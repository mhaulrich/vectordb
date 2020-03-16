import sys
from flask import Flask, abort, Response
from milvus import Milvus, IndexType, MetricType
import psycopg2
import json
from flask import request
import cProfile as profile
from VectorUtils import hash_vector


# These are the correct values for using with docker-compose
# If running outside docker-compose run with
# python3 app.py <SQL HOST> <SQL PORT>  <MILVUS_HOST> <MILVUS PORT>
MILVUS_HOST = 'milvus'
MILVUS_PORT = '19530'  # default value

POSTGRES_HOST = 'db'
POSTGRES_PORT = '5432'


if len(sys.argv) == 4:
    POSTGRES_HOST = sys.argv[0]
    POSTGRES_PORT = sys.argv[1]
    MILVUS_HOST = sys.argv[2]
    MILVUS_PORT = sys.argv[3]


# In outer section of code
pr = profile.Profile()
pr.disable()
app = Flask(__name__)

connection = None
global_milvus = None

metatable_name = 'vectordb_meta'

_INDEX_FILE_SIZE = 32  # max file size of stored index
_METRIC_TYPE = MetricType.IP

# In order not to connect to DB to check dimentions every time - we keep a cache
# For now - we assume that we will never have so many dbs that it is a problem to have this in memory
db_dimensions_cache = {}


def abort_missing_parameters(method, need_to_have_parameters, additional_comment=None):
    error_string = 'ERROR:  Missing parameters. Method "' + method + "' needs the following parameters: " + ','.join(need_to_have_parameters) + '\n'
    if additional_comment:
        error_string = error_string + '\t' + additional_comment + '\n'
    abort(Response(error_string))


def abort_wrong_dimenions(dbname, reqqest_dims, db_dimensions):
    error_string = 'ERROR: vectors in ' + dbname + ' have ' + str(db_dimensions) + ' dimensions. Vector in request has ' + str(reqqest_dims) + '\n'
    abort(Response(error_string))


def connect_db():
    try:
        connection = psycopg2.connect(user='postgres',
                                      password='mysecretpassword',
                                      host=POSTGRES_HOST,
                                      port=POSTGRES_PORT,
                                      database='postgres')

        print(connection.get_dsn_parameters(), "\n")

        return connection

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)


# Get postgres connection
def get_connection():
    global connection
    if connection is None:
        connection = connect_db()
    return connection


# Create the meta table in postgres
# This holds information about what tables there are, dims in table and index type
def create_meta_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE " + metatable_name + """ 
          (
                   name VARCHAR(30) NOT null,
                   dims INT NOT null,
                   index_type VARCHAR(30),
                   PRIMARY KEY(name)
          )
          """)
    cursor.close()
    conn.commit()


# Create new vector table in postgres
# This creates the table and also adds information about it to the meta table
def create_vector_table_in_db(name, dims, index_type):
    conn = get_connection()
    cursor = conn.cursor()
    # Write info about new vector table in meta
    cursor.execute("INSERT INTO " + metatable_name + " (name, dims, index_type) VALUES (%s, %s, %s)",
                   (name, dims, index_type))

    # Create table in sql db to map from vector hashes to ids
    cursor.execute("CREATE TABLE " + name + """ 
              (
                   vector_hash BIGINT NOT null,
                   asset_id VARCHAR(20) NOT null,
                   PRIMARY KEY(vector_hash, asset_id)
              )
              """)
    cursor.close()
    conn.commit()


# Check if table exists in postgres
def check_table_exists(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM pg_catalog.pg_tables WHERE tablename = '" + table_name + "'")
        cursor.close()
        conn.commit()
        if cursor.rowcount < 1:
            return False
        else:
            return True
    except (Exception, psycopg2.Error) as e:
        # Don't really like this. We must be able to check if the error is actually that the table does not exist
        print('Table does not exist')
        cursor.close()
        return False;


# At every startup we check if the metatable exists in postgres. If not - create it
# Hopefully it only create the first time the vector db is started
def init_db():
    meta_table_exists = check_table_exists(metatable_name)
    if not meta_table_exists:
        print('No metatable, creating it')
        create_meta_table()
    else:
        print('Meta table already exists')


def check_table_exists_milvus(table_name):
    milvus = get_milvus()

    status, ok = milvus.has_table(table_name)
    return ok


# Create new table in out vector db. This consists of two thiings
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


def get_db_dimensions(dbname):
    dims = db_dimensions_cache.get(dbname)
    if dims is None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT dims FROM " + metatable_name + ' WHERE name = %s', (dbname,))
        row = cursor.fetchone()
        dims = row[0]
        db_dimensions_cache[dbname] = dims

    return dims


def show_rows():
    cursor = connection.cursor()
    # Print PostgreSQL Connection properties
    print(connection.get_dsn_parameters(), "\n")

    cursor.execute("SELECT * FROM " + metatable_name)
    print("The number of parts: ", cursor.rowcount)
    row = cursor.fetchone()

    i = 1
    while row is not None:
        print(row)
        row = cursor.fetchone()
        i = i + 1
    return i


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


# Check if what is in meta corresponds with what vector_hash tables there are, and what there is in milvus
@app.route('/checkintegrity')
def check_db_integrety():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM " + metatable_name)
    print("Number of tables in meta: ", cursor.rowcount)
    row = cursor.fetchone()

    all_good = True

    print('Table name\tExists in postgres\tExists in Milvus')
    while row is not None:
        table_name = row[0]
        exists_in_postgres = check_table_exists(table_name)
        exists_in_milvus = check_table_exists_milvus(table_name)
        print(table_name + '\t\t' + str(exists_in_postgres) + '\t\t\t\t' + str(exists_in_milvus))

        all_good = all_good and exists_in_postgres and exists_in_milvus

        row = cursor.fetchone()

    if all_good:
        return 'All ok'
    else:
        return 'DB integrety bad. Check server output for details'


@app.route('/')
def hello():
    count = show_rows()
    return 'Total rows: ' + str(count)


@app.route('/createdb')
def create_new_db():
    db_name = request.args.get('dbname')
    dimensions = int(request.args.get('dimensions'))
    (error_code, return_string) = create_vector_db(db_name, dimensions)
    return return_string


@app.route('/deletedb')
def delete_db():
    all_ok = True
    db_name = request.args.get('dbname')
    # First let ud delete in Milvus
    milvus = get_milvus()
    status = milvus.delete_table(db_name)
    if status.code != 0:
        all_ok = False
        print(status.message)

    # Now let us delete in postgresdb - vector table first
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('DROP TABLE ' + db_name)
    except (Exception, psycopg2.Error) as error:
        all_ok = False
        print("Error while deleting table from postgres", error)
    cursor.close()
    cursor = conn.cursor()

    try:
        cursor.execute('DELETE FROM ' + metatable_name + ' WHERE name = %s', (db_name,))
    except (Exception, psycopg2.Error) as error:
        all_ok = False
        print("Error while trying to delete table in metatable.", error)
    cursor.close()

    conn.commit()

    if all_ok:
        return 'Table deleted'
    else:
        return 'Problem with deleting table. See server log for details'


# Tries to insert vectorhashes and asset_ids into postgres
# Note that the first check is done on at a time so that we know for later whether or not the vector should be inserted into milvus
# There are three cases:
# 1. Vector hash and asset_id already exists in db\
#    Nothing should be done
# 2. Vector hash exists but this asset_id does not
#    (Vector hash, asset_id) should be inserted into db
#    No vector should be added to Milvus as it is already there
# 3. New vector hash.
#    (Vector hash, asset_id) should be inserted into db
#    Vector should be inserted into milvus

def insert_vectorhash(dbname, vector_hashes, asset_ids):
    insert_vector_into_milvus = []

    # Create list of tuples to use in sql
    list_of_both = []
    # First check if vector hashes exist already
    conn = get_connection()
    cursor = conn.cursor()
    for i in range(len(vector_hashes)):
        vector_hash = vector_hashes[i]
        asset_id = asset_ids[i]
        list_of_both.append((vector_hash, asset_id))

        try:
            cursor.execute('SELECT * FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
        except (Exception, psycopg2.Error) as error:
            print("DB does not exist", error)
            conn.rollback()
            return

        insert_vector_into_milvus.append(cursor.rowcount == 0)

    # The actual inserts we do all at once
    # Because of primary key contraints we can simple add the vector_hash asset_id pair to the db
    # the DB will not add it if it exsists already
    try:
        cursor.executemany('INSERT INTO ' + dbname + ' (vector_hash, asset_id) VALUES (%s, %s) ON CONFLICT DO NOTHING', list_of_both)
    except (Exception, psycopg2.Error) as error:
        print("Row already exists in db - this is ok", error)
        conn.rollback()

    cursor.close()
    conn.commit()

    return insert_vector_into_milvus


def insert_into_milvus(db_name, vector_hashes, vectors):
    milvus = get_milvus()

    # Milvus expets lists
    milvus.insert(db_name, records=vectors, ids=vector_hashes)


@app.route('/insertvector', methods=['PUT'])
def insert_vector():
    pr.enable()
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    if (db_name is None) or (vector_with_id is None):
        abort_missing_parameters('insert', ['dbname'], 'and vector in body')

    asset_id = vector_with_id['name']
    vector = vector_with_id['vector']
    query_vector_dims = len(vector)
    db_dims = get_db_dimensions(db_name)
    if query_vector_dims != db_dims:
        abort_wrong_dimenions(db_name, query_vector_dims, db_dims)

    vector_hash = hash_vector(vector)

    vector_should_be_added_to_milvus = insert_vectorhash(db_name, [vector_hash], [asset_id])
    if vector_should_be_added_to_milvus[0]:
        insert_into_milvus(db_name, vector_hash, vector)

    pr.disable()
    return 'hej'


@app.route('/insertvectors', methods=['PUT'])
def insert_vectors():
    pr.enable()
    db_name = request.args.get('dbname')
    vectors_with_ids = request.json
    if (db_name is None) or (vectors_with_ids is None):
        abort_missing_parameters('insert', ['dbname'], 'and vectors in body')

    db_dims = get_db_dimensions(db_name)
    asset_ids = []
    vectors = []
    vector_hashes = []

    for vector_with_id in vectors_with_ids:
        asset_id = vector_with_id['name']
        vector = vector_with_id['vector']
        query_vector_dims = len(vector)
        if query_vector_dims != db_dims:
            abort_wrong_dimenions(db_name, query_vector_dims, db_dims)
        vector_hash = hash_vector(vector)

        asset_ids.append(asset_id)
        vectors.append(vector)
        vector_hashes.append(vector_hash)
    vectors_should_be_added_to_milvus = insert_vectorhash(db_name, vector_hashes, asset_ids)

    # Find hashes and vectors that should be added to milvus
    # We do this in a dict to avoid actually adding the same vector more than once
    milvus_vectors = {}
    for i in range(len(vectors_should_be_added_to_milvus)):
        if vectors_should_be_added_to_milvus[i]:
            milvus_vectors[vector_hashes[i]] = vectors[i]

    print('Adding: ' + str(len(milvus_vectors)) + ' to Milvus')

    if len(milvus_vectors) > 0:
        insert_into_milvus(db_name, list(milvus_vectors.keys()), list(milvus_vectors.values()))

    pr.disable()
    return 'hej'


def get_assets(dbname, vector_hash):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT asset_id FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
    asset_ids = []
    if cursor.rowcount > 0:
        row = cursor.fetchone()
        while row is not None:
            asset_id = row[0]
            asset_ids.append(asset_id)
            row = cursor.fetchone()
    return asset_ids


@app.route('/lookupexact', methods=['PUT'])
def lookup_exact():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)
    assets_ids = get_assets(db_name, vector_hash)
    assets_ids_json = json.dumps(assets_ids)
    return assets_ids_json


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


@app.route('/lookup', methods=['PUT'])
def lookup():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)
    assets_ids = get_assets(db_name, vector_hash)

    results = []

    # Find exact mataches
    exact_matches = {'distance': 0, 'asset_ids': assets_ids}
    results.append(exact_matches)

    milvus_results = lookup_milvus(db_name, vector, 20)
    for milvus_res in milvus_results:
        vector_hash = milvus_res['vectorhash']
        distance = milvus_res['distance']
        asset_ids = get_assets(db_name, vector_hash)
        this_result = {'distance': distance, 'asset_ids': asset_ids}
        results.append(this_result)

    results_json = json.dumps(results)
    return results_json


@app.route('/listdbs', methods=['GET'])
def listdba():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM " + metatable_name)
    row = cursor.fetchone()
    tables = []

    while row is not None:
        table_name = row[0]
        tables.append(table_name)
        row = cursor.fetchone()

    milvus = get_milvus()

    table_infos = []
    for table_name in tables:
        status, milvus_table = milvus.describe_table(table_name)
        dims = milvus_table.dimension
        index_file_size = milvus_table.index_file_size
        metric_type = milvus_table.metric_type
        status, num_rows = milvus.get_table_row_count(table_name)
        cursor.execute("SELECT count(*) FROM " + table_name)
        row = cursor.fetchone()
        n_assets = row[0]
        table_info = {'name': table_name, 'dimensions': dims, 'metric_type': str(metric_type), 'no_vectors': num_rows, 'no_assets' : n_assets}
        table_infos.append(table_info)

    cursor.close()
    conn.commit()
    return json.dumps(table_infos)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    pr.dump_stats('profile.pstat')
    shutdown_server()
    return 'Server shutting down...'

init_db()

if __name__ == "__main__":
    app.run(port=5001)
