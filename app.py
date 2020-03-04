import sys
from flask import Flask
from milvus import Milvus, IndexType, MetricType
import psycopg2
import hashlib
import json
from flask import request

app = Flask(__name__)

connection = None
milvus = None


metatable_name = 'vectordb_meta'

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value
_INDEX_FILE_SIZE = 32  # max file size of stored index
_METRIC_TYPE = MetricType.IP


def connect_db():
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="mysecretpassword",
                                      host="localhost",
                                      port="5432",
                                      database="postgres")

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
    cursor.execute("INSERT INTO " + metatable_name + " (name, dims, index_type) VALUES (%s, %s, %s)", (name, dims, index_type))

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
    param = {'host': _HOST, 'port': _PORT}
    status = milvus.connect(**param)
    if status.OK():
        return milvus
    else:
        print("Server connect fail.")
        sys.exit(1)


# Get milvus connection
def get_milvus():
    global milvus
    if milvus is None:
        milvus = connect_to_milvus()
    return milvus


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


# Tries to insert vectorhash and asset_id into postgres
# There are three cases:
# 1. Vector hash and asset_id already exists in db\
#    Nothing should be done
# 2. Vector hash exists but this asset_id does not
#    (Vector hash, asset_id) should be inserted into db
#    No vector should be added to Milvus as it is already there
# 3. New vector hash.
#    (Vector hash, asset_id) should be inserted into db
#    Vector should be inserted into milvus

def insert_vectorhash(dbname, vector_hash, asset_id):

    insert_vector_into_milvus = True

    # First check if vector hash exists already
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash, ))
    if cursor.rowcount > 0:
        insert_vector_into_milvus = False

    print(vector_hash)
    print(asset_id)

    # Because of primary key contraints we can simple add the vector_hash asset_id pair to the db
    # the DB will not add it if it exsists already
    try:
        cursor.execute('INSERT INTO ' + dbname + ' (vector_hash, asset_id) VALUES (%s, %s)', (vector_hash, asset_id))

        print(cursor.query)
    except (Exception, psycopg2.Error) as error:
        print("Row already exists in db - this is ok", error)

    cursor.close
    conn.commit()

    return insert_vector_into_milvus


def insert_into_milvus(db_name, vector_hash, vector):
    milvus = get_milvus()

    # Milvus expets lists
    ids = [vector_hash]
    vector_list = [vector]

    print(type(ids))
    print(ids)

    milvus.insert(db_name, records=vector_list, ids=ids)


def hash_vector(vector):
    vector_str = ','.join(['%.5f' % num for num in vector])
    m = hashlib.md5()
    m.update(bytes(vector_str, encoding='utf-8'))
    md5_bytes = m.digest()
    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


@app.route('/insert', methods = ['PUT'])
def insert_vector():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    asset_id = vector_with_id['name']
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)

    vector_should_be_added_to_milvus = insert_vectorhash(db_name, vector_hash, asset_id)
    if vector_should_be_added_to_milvus:
        insert_into_milvus(db_name, vector_hash, vector)

    return 'hej'


def get_assets(dbname, vector_hash):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT asset_id FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
    print(cursor.query)
    asset_ids = []
    if cursor.rowcount > 0:
        row = cursor.fetchone()
        while row is not None:
            asset_id = row[0]
            asset_ids.append(asset_id)
            row = cursor.fetchone()
    return asset_ids


@app.route('/lookupexact', methods = ['PUT'])
def lookup_exact():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)
    print(str(vector_hash))
    assets_ids = get_assets(db_name, vector_hash)
    assets_ids_json = json.dumps(assets_ids)
    return assets_ids_json


def lookup_milvus(dbname, vector, k=10):
    milvus = get_milvus()

    vector_list = [vector]

    param =  {
        'table_name': dbname,
        'query_records': vector_list,
        'top_k': k,
        'nprobe' : 10
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


@app.route('/lookup', methods = ['PUT'])
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

    milvus_results = lookup_milvus(db_name, vector, 10)
    for milvus_res in milvus_results:
        vector_hash = milvus_res['vectorhash']
        distance = milvus_res['distance']
        asset_ids = get_assets(db_name, vector_hash)
        this_result = {'distance': distance, 'asset_ids': assets_ids}
        results.append(this_result)

    results_json = json.dumps(results)
    return results_json





init_db()

app.run(port=5001)
