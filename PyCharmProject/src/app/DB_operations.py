import psycopg2

connection = None

POSTGRES_HOST = 'db'
POSTGRES_PORT = '5432'

METATABLE_NAME = 'vectordb_meta'

# In order not to connect to DB to check dimentions every time - we keep a cache
# For now - we assume that we will never have so many dbs that it is a problem to have this in memory
db_dimensions_cache = {}


def setPostGresHost(postgres_host, postgres_port):
    global POSTGRES_HOST
    global POSTGRES_PORT
    POSTGRES_HOST = postgres_host
    POSTGRES_PORT = postgres_port


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
    cursor.execute("CREATE TABLE " + METATABLE_NAME + """ 
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
    cursor.execute("INSERT INTO " + METATABLE_NAME + " (name, dims, index_type) VALUES (%s, %s, %s)",
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
    meta_table_exists = check_table_exists(METATABLE_NAME)
    if not meta_table_exists:
        print('No metatable, creating it')
        create_meta_table()
    else:
        print('Meta table already exists')


def get_db_dimensions(dbname):
    dims = db_dimensions_cache.get(dbname)
    if dims is None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT dims FROM " + METATABLE_NAME + ' WHERE name = %s', (dbname,))
        row = cursor.fetchone()
        dims = row[0]
        db_dimensions_cache[dbname] = dims

    return dims


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
