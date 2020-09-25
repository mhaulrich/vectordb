from milvus import Milvus, IndexType, MetricType, NotConnectError
import numpy as np
import sys
import os

LIMIT_RETRIES = 10
DEBUG = os.environ.get('VECTORDB_DEBUG') == 'true'

DEFAULT_INDEX_SIZE = 1024
DEFAULT_METRIC = MetricType.L2

class VectorIndex:
    """A class for communication with Milvus vector index"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._indexFileSize = DEFAULT_INDEX_SIZE
        self._metricType = DEFAULT_METRIC
        self._milvus =  None

        self.init()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def connect(self):

        param = {'host': self.host, 'port': self.port}
        if DEBUG:
            print("Connecting to Milvus, %s:%s"%(self.host,self.port))

        retries = 0
        while retries < LIMIT_RETRIES:
            try:
                self._milvus = Milvus(host=self.host, port=self.port)
                if DEBUG:
                    print("Successfully connected to Milvus")
                return self._milvus
            except Exception as err:
                print("Failed to connect to Milvus: %s"%str(err))
            retries += 1
            print("Retrying %d"%retries)
        else:
            print("Server connect fail.")
            sys.exit(1)

    def milvus(self):
        if self._milvus is None:
            self.connect()
        return self._milvus

    def init(self):
        self.connect()

    def tableExists(self, tableName):
        """Check if a vector table/index exists"""
        milvus = self.milvus()
        status, ok = milvus.has_collection(tableName)
        return ok

    def createTable(self, tableName, dimensions, index_type):
        """Create new table/index in the vector db of the provided dimensionality and type"""
        if DEBUG:
            print("Creating vector database with name '%s', %d dimensions and type '%s'."%(tableName, dimensions,index_type))

        if self.tableExists(tableName):
            raise VectorIndexError("Table already exists")
        else:
            milvus = self.milvus()
            param = {
                'collection_name': tableName,
                'dimension': dimensions,
                'index_file_size': self._indexFileSize,
                'metric_type': self._metricType
            }
            table_status = milvus.create_collection(param)
            if not table_status.OK():
                raise VectorIndexError("Could not create table '%s': %s."%(tableName,table_status.message))

            index_param = {
                'nlist': 16384
            }
            index_status = milvus.create_index(tableName, index_type, index_param)
            if not index_status.OK():
                raise VectorIndexError("Could not create index: %s."%index_status.message)

    def flushTable(self, tableName):
        """Flush a table to disk. This must be done before points are available
        for lookup"""
        milvus = self.milvus()
        milvus.flush([tableName])

    def deleteTable(self, tableName):
        milvus = self.milvus()
        status = milvus.drop_collection(tableName)
        if not status.OK():
            raise VectorIndexError("Could not delete table '%s': %s."%(tableName,status.message))


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
            raise VectorIndexError("Could not insert: %s"%status.message)
        return ids

    def lookup(self, tableName, vector, k=10):
        """Lookup the 'k' nearest neighbours to the query vectors."""
        milvus = self.milvus()
        vector_list = self.make2DFloat(vector).tolist()
        print("Looking up %d nearest neighbours in table '%s' for query points: %s"%(k,tableName,vector))

        params = {
            'nprobe': 16, #IVF_Flat
            'search_length': 50, #RNSG
            'ef': max(k,2000), # HNSW
            'search_k': -1 # ANNOY - 5% of data
        }
        status, queryResults = milvus.search(tableName, k, vector_list, params=params)
        if not status.OK():
            raise VectorIndexError("Could not lookup: %s"%status.message)
        resultsArr = []

        for id_list, dis_list in zip(queryResults.id_array, queryResults.distance_array):
            neighbourResults = []
            for nid, distance in zip(id_list, dis_list):
                neighbourResults.append({
                    'distance': distance,
                    'id': str(nid)
                    })
            resultsArr.append(neighbourResults)
        return resultsArr

    def describeTables(self, tables):
        """Describe the tables specified by 'tables'."""
        returnAsList = True
        if type(tables) == str:
            tables = [tables]
            returnAsList = False
        milvus = self.milvus()
        table_infos = []
        for table_name in tables:
            status, milvus_table = milvus.get_collection_info(table_name)
            if not status.OK():
                raise VectorIndexError("Could not describe table '%s'': %s"%(table_name,status.message))
            status, milvus_idx = milvus.get_index_info(table_name)
            if not status.OK():
                raise VectorIndexError("Could not describe index '%s'': %s"%(table_name,status.message))
            # index_file_size = milvus_table.index_file_size
            status, num_rows = milvus.count_entities(table_name)
            if not status.OK():
                raise VectorIndexError("Could not get number of rows for table '%s'': %s"%(table_name,status.message))
            table_info = {
                'name': table_name,
                'dimensions': milvus_table.dimension,
                'no_vectors': num_rows,
                'metric_type': str(milvus_table.metric_type),
                'index': {
                        'type': str(milvus_idx.index_type),
                        'size': milvus_table.index_file_size,
                        'params': milvus_idx.params
                    }
                }
            table_infos.append(table_info)
        if returnAsList:
            return table_infos
        else:
            return table_infos[0]

    def describePoint(self, tableName, pointID):
        milvus = self.milvus()
        _, point = milvus.get_entity_by_id(tableName, [pointID,])
        return point[0]

    def make2DFloat(self, vector):
        vector = np.array(vector, dtype=float)
        if vector.ndim == 1: #Make sure it's always 2D
            vector = vector.reshape((1,-1))
        return vector

class VectorIndexError(Exception):
    def __init__(self, message, suberror=None):
        self.message = message
        self.suberror = suberror
    def __str__(self):
        if self.suberror:
            return 'VectorIndexError: %s. Suberror: %s. '%(self.message, self.suberror)
        else:
            return 'VectorIndexError: %s. '%(self.message)
