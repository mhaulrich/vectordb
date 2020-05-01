from milvus import Milvus, IndexType, MetricType, NotConnectError
import numpy as np
import sys
import os

MILVUS_HOST = 'milvus'
MILVUS_PORT = '19530'  # default value

_INDEX_FILE_SIZE = 32  # max file size of stored index
_METRIC_TYPE = MetricType.IP

LIMIT_RETRIES = 10
DEBUG = os.environ.get('VECTORDB_DEBUG') == 'true'

global_milvus = None


class VectorIndex:
    """A class for communication with Milvus vector index"""
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._indexFileSize = 32
        self._metricType = MetricType.IP
        self._milvus =  None
        
        self.init()
        
    def __del__(self):
        try:
            self.close()
        except:
            pass
        
    def connect(self):
        self._milvus = Milvus()
        param = {'host': self.host, 'port': self.port}
        if DEBUG:
            print("Connecting to Milvus, %s:%s"%(self.host,self.port))
        
        retries = 0
        while retries < LIMIT_RETRIES:
            try:
                status = self._milvus.connect(**param)
                if status.OK():
                    if DEBUG:
                        print("Successfully connected to Milvus")
                    return self._milvus
                print("Bad Milvus status code: %s"%str(status))
            except NotConnectError as err:
                print("Failed to connect to Milvus: %s"%str(err))
            retries += 1
            print("Retrying %d"%retries)
        else:
            print("Server connect fail.")
            sys.exit(1)
        
    def milvus(self):
        if self._milvus is None or not self._milvus.connected():
            self.connect()
        return self._milvus
        
    def init(self):
        self.connect()
        
    def tableExists(self, tableName):
        """Check if a vector table/index exists"""
        milvus = self.milvus()
        status, ok = milvus.has_table(tableName)
        return ok 


    # 
    def createTable(self, tableName, dimensions, index_type):
        """Create new table/index in the vector db of the provided dimensionality and type"""
        if DEBUG:
            print("Creating vector database with name '%s', %d dimensions and type '%s'."%(tableName, dimensions,index_type))
    
        if self.tableExists(tableName):
            raise VectorIndexError("Table already exists")
        else:
            milvus = self.milvus()
            param = {
                'table_name': tableName,
                'dimension': dimensions,
                'index_file_size': self._indexFileSize,
                'metric_type': self._metricType
            }
            table_status = milvus.create_table(param)
            if not table_status.OK():
                raise VectorIndexError("Could not create table '%s': %s."%(tableName,table_status))
    
            index_param = {
                'index_type': index_type,
                'nlist': 2048
            }
            index_status = milvus.create_index(tableName, index_param)
            if not index_status.OK():
                raise VectorIndexError("Could not create index: %s."%index_status)

    def deleteTable(self, tableName):
        milvus = self.milvus()
        status = milvus.delete_table(tableName)
        if not status.OK():
            raise VectorIndexError("Could not delete table '%s': %s."%(tableName,status))


    def insert(self, tableName, vectors, ids=None):
        """Insert vectors into milvus using the associated ids. Both vectors 
        and ids must be of type array/list. If ids is None, ids are automatically
        assigned.
        Returns a list of ids of the vectors inserted."""
        vectors = self.make2DFloat(vectors).tolist()
        if DEBUG:
            print("Inserting vectors into table '%s':\n%s\nwith ids:%s"%(tableName, vectors, ids))
        milvus = self.milvus()
        status, ids = milvus.insert(tableName, records=vectors, ids=ids)
        if not status.OK():
            raise VectorIndexError("Could not insert: %s"%status)
        return ids

    def lookup(self, tableName, vector, k=10):
        """Lookup the 'k' nearest neighbours to the query vectors."""
        milvus = self.milvus()
        vector_list = self.make2DFloat(vector).tolist()
        print("Looking up %d nearest neighbours in table '%s' for query points: %s"%(k,tableName,vector))
        param = {
            'table_name': tableName,
            'query_records': vector_list,
            'top_k': k,
            'nprobe': 10
        }
    
        status, queryResults = milvus.search_vectors(**param)
        if not status.OK():
            raise VectorIndexError("Could not lookup: %s"%status)
        resultsArr = []
        for queryResult in queryResults:
            known_results = {}
            neighbourResults = []
            for neighbour in queryResult:
                if neighbour.id in known_results:
                    continue
                known_results[neighbour.id] = 1
                neighbourResults.append({
                    'distance': neighbour.distance, 
                    'id': neighbour.id
                    })
            resultsArr.append(neighbourResults)
        return resultsArr
    
    def describeTables(self, tables):
        """Describe the tables specified by 'tables'."""
        returnList = True
        if type(tables) == str:
            tables = [tables]
            returnList = False
        milvus = self.milvus()
        table_infos = []
        for table_name in tables:
            status, milvus_table = milvus.describe_table(table_name)
            if not status.OK():
                raise VectorIndexError("Could not describe table '%s'': %s"%(table_name,status))        
            dims = milvus_table.dimension
            # index_file_size = milvus_table.index_file_size
            metric_type = milvus_table.metric_type
            status, num_rows = milvus.count_table(table_name)
            if not status.OK():
                raise VectorIndexError("Could not get number of rows for table '%s'': %s"%(table_name,status))
            table_info = {'name': table_name, 'dimensions': dims, 'metric_type': str(metric_type), 'no_vectors': num_rows}
            table_infos.append(table_info)
        if returnList:
            return table_infos
        else:
            return table_infos[0]
    
    def describePoint(self, tableName, pointID):
        milvus = self.milvus()
        return milvus.get_vector_by_id(tableName, pointID)
        
    def make2DFloat(self, vector):
        vector = np.array(vector, dtype=float)
        if vector.ndim == 1: #Make sure it's always 2D
            vector = vector.reshape((1,-1))
        return vector

class VectorIndexError(Exception):
    def __init__(self, message, suberror=None):
        self.message = None
        self.suberror = None
    def __str__(self):
        if self.suberror:
            return 'VectorIndexError: %s. Suberror: %s. '%(self.message, self.suberror)
        else:
            return 'VectorIndexError: %s. '%(self.message)