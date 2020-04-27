import psycopg2
import time
import os


METATABLE_NAME = 'vectordb_meta'
LIMIT_RETRIES = 10
DEBUG = os.environ.get('VECTORDB_DEBUG') == 'true'

class AssetDatabase:
    """A class for communication with postgres asset database"""
    
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        
        self._connection = None
        self._cursor = None
        
        # In order not to connect to DB to check dimentions every time - we keep a cache
        # For now - we assume that we will never have so many dbs that it is a problem to have this in memory
        self._dimensions_cache = {}
        
        self.init()
        
    def __del__(self):
        try:
            self.close()
        except:
            pass
        
    def _retryOperation(self, operation, operationArgs=[], operationKwargs={}, sleepTime=5, failCallback=None):
        retries = 0
        while retries < LIMIT_RETRIES:
            try:
                return operation(*operationArgs, **operationKwargs)
            except (psycopg2.DatabaseError, psycopg2.OperationalError) as error:
                if retries >= LIMIT_RETRIES:
                    raise error
                else:
                    retries += 1
                    print("Error executing Postgres operation %s: %s."%(operation, str(error).strip()))
                    print("Retrying %d"%retries)
                    time.sleep(sleepTime)
                    if failCallback is not None:
                        failCallback()
            except (Exception, psycopg2.Error) as error:
                raise error
                        
        
    def connect(self):
        """Connect to Postgres database, retrying every 3rd second for up to 10 times."""
        if not self._connection or self._connection.closed:
            if DEBUG:
                print("Connecting to Postgres..")
            connectArgs = {'user': self.user, 
                           'password': self.password, 
                           'host': self.host, 
                           'port': self.port, 
                           'database': self.database, 
                           'connect_timeout': 3}
            self._connection = self._retryOperation(psycopg2.connect, operationKwargs=connectArgs)
            if DEBUG:
                print("Successfully connected")
                
    def cursor(self):
        """Get the current cursor"""
        if not self._cursor or self._cursor.closed:
            if not self._connection:
                self.connect()
            self._cursor = self._connection.cursor()
        return self._cursor
    
    def commit(self):
        """Commit any pending transactions"""
        self._connection.commit()
        
    def rollback(self):
        """Rollback any non-committed transactions"""
        self._connection.rollback()
                    
    def retryingExecute(self, query, queryVars=None):
        """Carry out a cursor.execute, retrying upon failure every second for up to 10 times"""
        cursor = self.cursor()
        if DEBUG:
            print("Executing \"%s\" with params: %s"%(query,queryVars))
        self._retryOperation(cursor.execute, [query, queryVars], sleepTime=1, failCallback=self.reset)
        return cursor
        
    def reset(self):
        """Close connection and reconnect"""
        self.close()
        self.connect()
        
    def close(self):
        """Close connection"""
        if self._connection:
            self._connection.close()
            print("PostgreSQL connection is closed")
        self._connection = None
        
    # ------------- AssetDatabase specific methods ------------------------
        
    def init(self):
        """Initialize AssetDatabase"""
        #Setup connection
        self.connect()
        
        #Create metadata table if missing
        if self.tableExists(METATABLE_NAME):
            if DEBUG:
                print('Meta table already exists')
        else:
            if DEBUG:
                print('No metatable, creating it')
            self.createMetaTable()
   

    def createMetaTable(self):
        """Create the meta table in postgres.
        This holds information about what tables there are, dims in table and index type"""
        cursor = self.cursor()
        cursor.execute("CREATE TABLE " + METATABLE_NAME + """ 
              (
                       name VARCHAR(30) NOT null,
                       dims INT NOT null,
                       index_type VARCHAR(30),
                       PRIMARY KEY(name)
              )
              """)
        cursor.close()
        self.commit()
    
    def createVectorTable(self, name, dims, index_type):
        """Create new vector table in postgres.
        This creates the table and also adds information about it to the meta table"""
        cursor = self.cursor()
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
        self.commit()
        
    def deleteVectorTable(self, name):
        """Deletes a vector table in postgres.
        This removes the table and also removes it's metadata from teh meta table."""
        cursor = self.cursor()
        # Delete table
        cursor.execute('DROP TABLE ' + name)
        # Delete meta info
        cursor.execute('DELETE FROM ' + METATABLE_NAME + ' WHERE name = %s', (name,))
        cursor.close()
        self.commit()
    
    
    # Check if table exists in postgres
    def tableExists(self, table_name):
        cursor = self.cursor()
        try:
            cursor.execute("SELECT * FROM pg_catalog.pg_tables WHERE tablename = '" + table_name + "'")
            cursor.close()
            self.commit()
            if cursor.rowcount < 1:
                return False
            else:
                return True
        except (Exception, psycopg2.Error) as e:
            # Don't really like this. We must be able to check if the error is actually that the table does not exist
            print('Table does not exist:', e)
            cursor.close()
            return False
    
    def getDimensions(self, dbname):
        """Get the dimensionality of a vectorTable, e.g. 3 for a 3D table. This is not the number of rows"""
        dims = self._dimensions_cache.get(dbname)
        if dims is None:
            cursor = self.cursor()
            cursor.execute("SELECT dims FROM " + METATABLE_NAME + ' WHERE name = %s', (dbname,))
            row = cursor.fetchone()
            cursor.close()
            dims = row[0]
            self._dimensions_cache[dbname] = dims
    
        return dims
    
    def getExistingTables(self):
        """Returns a list of the names of all existing tables registered in the Meta-table."""
        cursor = self.cursor()
        cursor.execute("SELECT name FROM " + METATABLE_NAME)

        tableNames = []
        for row in cursor:
            tableNames.append(row[0])
        cursor.close()
        return tableNames
    
    def getNumberOfAssets(self, tablename):
        """Returns the number (int) of assets 'tablename' contains. If tablename is of type
        List, a list of the number of assets for each table name in the list."""
        returnAsList = True
        if type(tablename) is not list:
            tablename = [tablename]
            returnAsList = False
        counts = []
        cursor = self.cursor()
        for table in tablename:
            cursor.execute("SELECT count(*) FROM " + table)
            row = cursor.fetchone()
            counts.append(row[0])
        cursor.close()
        if not returnAsList:
            return counts[0]
        else:
            return counts
    
    def insertVectorHashes(self, dbname, vector_hashes, asset_ids):
        """Tries to insert vectorhashes and asset_ids into postgres
        Note that the first check is done on at a time so that we
        know for later whether or not the vector should be inserted into milvus
        There are three cases:
            1. Vector hash and asset_id already exists in db\
                Nothing should be done
            2. Vector hash exists but this asset_id does not
                (Vector hash, asset_id) should be inserted into db
                No vector should be added to Milvus as it is already there
            3. New vector hash.
                (Vector hash, asset_id) should be inserted into db
                Vector should be inserted into milvus"""
        insert_vector_into_milvus = []
    
        # Create list of tuples to use in sql
        list_of_both = []
        # First check if vector hashes exist already
        cursor = self.cursor()
        for i, vector_hash in enumerate(vector_hashes):
            list_of_both.append((vector_hash, asset_ids[i]))
            try:
                cursor.execute('SELECT * FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
            except (Exception, psycopg2.Error) as error:
                print("DB does not exist", error)
                self.rollback()
                cursor.close()
                return
            insert_vector_into_milvus.append(cursor.rowcount == 0)
    
        # The actual inserts we do all at once
        # Because of primary key contraints we can simple add the vector_hash asset_id pair to the db
        # the DB will not add it if it exsists already
        # TODO: How does this handle if only some of the pairs exist?
        try:
            cursor.executemany('INSERT INTO ' + dbname + ' (vector_hash, asset_id) VALUES (%s, %s) ON CONFLICT DO NOTHING', list_of_both)
        except (Exception, psycopg2.Error) as error:
            print("Row already exists in db - this is ok", error)
            self.rollback()  
        cursor.close()
        self.commit()  
        return insert_vector_into_milvus
    
    
    def getAssets(self, dbname, vector_hash):
        cursor = self.cursor()
        cursor.execute('SELECT asset_id FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
        asset_ids = []
        for row in cursor:
            asset_ids.append(row[0])
        cursor.close()
        return asset_ids