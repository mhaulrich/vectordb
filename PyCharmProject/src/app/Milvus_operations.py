from milvus import Milvus, IndexType, MetricType
from DB_operations import *
import sys

MILVUS_HOST = 'milvus'
MILVUS_PORT = '19530'  # default value

_INDEX_FILE_SIZE = 32  # max file size of stored index
_METRIC_TYPE = MetricType.IP

global_milvus = None


def setMilvusHost(milvus_host, milvus_port):
    global MILVUS_HOST
    global MILVUS_PORT
    MILVUS_HOST = milvus_host
    MILVUS_PORT = milvus_port


def check_table_exists_milvus(table_name):
    milvus = get_milvus()

    status, ok = milvus.has_table(table_name)
    return ok


# Create new table in our vector db. This consists of two thiings
# - create table in postgres to hold mapping from vector hash to assets
# - create table in milvus for the actual vector search
def create_vector_db(vector_db_name, dimensions, index_type='IVFLAT'):
    if check_table_exists(vector_db_name):
        return -1, 'Error: Vector database with name: ' + vector_db_name + ' already exists'
    create_vector_table_in_db(vector_db_name, dimensions, index_type)

    milvus = get_milvus()

    if check_table_exists_milvus(vector_db_name):
        return -1, 'Error: table with name: ' + vector_db_name + ' exists in milvus (but not in postgres)'
    else:
        param = {
            'table_name': vector_db_name,
            'dimension': dimensions,
            'index_file_size': _INDEX_FILE_SIZE,
            'metric_type': _METRIC_TYPE
        }
        milvus.create_table(param)

        index_param = {
            'index_type': IndexType.IVFLAT,
            'nlist': 2048
        }
        milvus.create_index(vector_db_name, index_param)

    return 1, 'Vector database with name: ' + vector_db_name + ' created'


def connect_to_milvus():
    milvus = Milvus()

    # Connect to Milvus server
    # You may need to change _HOST and _PORT accordingly
    param = {'host': MILVUS_HOST, 'port': MILVUS_PORT}
    status = milvus.connect(**param)
    if status.OK():
        return milvus
    else:
        print("Server connect fail.")
        sys.exit(1)


# Get milvus connection
def get_milvus():
    global global_milvus
    if global_milvus is None:
        global_milvus = connect_to_milvus()
    return global_milvus


def insert_into_milvus(db_name, vector_hashes, vectors):
    milvus = get_milvus()

    # Milvus expets lists
    milvus.insert(db_name, records=vectors, ids=vector_hashes)


def lookup_milvus(dbname, vector, k=10):
    milvus = get_milvus()

    vector_list = [vector]

    param = {
        'table_name': dbname,
        'query_records': vector_list,
        'top_k': k,
        'nprobe': 10
    }

    status, results = milvus.search_vectors(**param)
    if len(results) > 0:

        res = results[0]
        return_res = []

        known_results = {}

        for r in res:
            print(r)
            vectorhash = r.id
            if vectorhash in known_results:
                continue
            known_results[vectorhash] = 1
            resline = {'distance': r.distance, 'vectorhash': vectorhash}
            return_res.append(resline)
        return return_res
    else:
        return []
