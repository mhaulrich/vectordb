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
                raise AssetDatabaseError("Could not execute Postgres operation", error)
                        
        
    def connect(self):
        """Connect to Postgres database, retrying every 3rd second for up to 10 times."""
        if not self._connection or self._connection.closed:
            if DEBUG:
                print("Connecting to Postgres, %s:%s"%(self.host,self.port))
            connectArgs = {'user': self.user, 
                           'password': self.password, 
                           'host': self.host, 
                           'port': self.port, 
                           'database': self.database, 
                           'connect_timeout': 3}
            self._connection = self._retryOperation(psycopg2.connect, operationKwargs=connectArgs)
            if DEBUG:
                print("Successfully connected to Postgres")
                
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
        This creates the table and also adds information about it to the meta table.
        This call must be followed by a commit() call to be confirmed or a
        rollback() call to be cancelled."""
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
            if cursor.rowcount < 1:
                return False
            else:
                return True
        except (Exception, psycopg2.Error) as e:
            # Don't really like this. We must be able to check if the error is actually that the table does not exist
            print('Table does not exist:', e)
            return False
        finally:
            cursor.close()
            self.commit()
    
    def getDimensions(self, dbname):
        """Get the dimensionality of a vectorTable, e.g. 3 for a 3D table. This is not the number of rows"""
        dims = self._dimensions_cache.get(dbname)
        if dims is None:
            cursor = self.cursor()
            try:
                cursor.execute("SELECT dims FROM " + METATABLE_NAME + ' WHERE name = %s', (dbname,))
                row = cursor.fetchone()
                dims = row[0]
                self._dimensions_cache[dbname] = dims
            except (Exception, psycopg2.Error) as e:
                print("Failed to get dimensions: ", e)
            cursor.close()
            self.commit()  
        return dims
    
    def getExistingTables(self):
        """Returns a list of the names of all existing tables registered in the Meta-table."""
        cursor = self.cursor()
        tableNames = []
        try:
            cursor.execute("SELECT name FROM " + METATABLE_NAME)
            for row in cursor:
                tableNames.append(row[0])
        except (Exception, psycopg2.Error) as e:
                print("Failed to get existing tables: ", e)
        cursor.close()
        self.commit()
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
        try:
            for table in tablename:
                cursor.execute("SELECT count(*) FROM " + table)
                row = cursor.fetchone()
                counts.append(row[0])
        except (Exception, psycopg2.Error) as error:
            print("Failed to get number of assets: %s"%error)        
        cursor.close()
        self.commit()
        if not returnAsList:
            return counts[0]
        else:
            return counts
    
    def insertVectorHashes(self, dbname, vector_hashes, assets):
        """Tries to insert vectorhashes and assets into postgres
        Returns a boolean list indicating whether the vector_hash already 
        exists. This is useful for determining whether or not a vector should 
        be inserted into vector index.
        This call must be followed by a commit() call to be confirmed or a
        rollback() call to be cancelled."""
        vector_hash_exists = []
    
        # Create list of tuples to use in sql
        list_of_both = []
        # First check if vector hashes exist already
        cursor = self.cursor()
        for i, vector_hash in enumerate(vector_hashes):
            list_of_both.append((vector_hash, assets[i]))
            try:
                cursor.execute('SELECT * FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
            except (Exception, psycopg2.Error) as error:
                print("DB does not exist", error)
                cursor.close()
                return
            vector_hash_exists.append(cursor.rowcount != 0)
    
        # The actual inserts we do all at once
        # Because of primary key contraints we can simple add the vector_hash asset_id pair to the db
        # the DB will not add it if it exsists already
        # TODO: How does this handle if only some of the pairs exist?
        try:
            cursor.executemany('INSERT INTO ' + dbname + ' (vector_hash, asset_id) VALUES (%s, %s) ON CONFLICT DO NOTHING', list_of_both)
        except (Exception, psycopg2.Error) as error:
            print("Row already exists in db - this is ok", error)
        cursor.close()        
        return vector_hash_exists
    
    
    def getAssets(self, dbname, vector_hash):
        """Return a list of assets associated with the provided vector_hash"""
        cursor = self.cursor()
        asset_ids = []
        try:
            cursor.execute('SELECT asset_id FROM ' + dbname + ' WHERE vector_hash = %s', (vector_hash,))
            for row in cursor:
                asset_ids.append(row[0])
        except (Exception, psycopg2.Error) as error:
            print("Failed to lookup asset: %s"%error)        
        cursor.close()
        self.commit()        
        return asset_ids
    
    def getPointsWithAsset(self, dbname, asset_id):
        """Return a list of points associated with the provided asset_id"""
        cursor = self.cursor()
        points = []
        try:
            cursor.execute("SELECT vector_hash, array_agg(asset_id) FROM %s WHERE vector_hash IN (SELECT vector_hash FROM %s WHERE asset_id = '%s') GROUP BY vector_hash ORDER BY vector_hash"%(dbname,dbname,asset_id))
            for row in cursor:
                points.append({'id': str(row[0]), 'assets': row[1]})
        except (Exception, psycopg2.Error) as error:
            print("Failed to lookup points for asset: %s"%error)        
        cursor.close()
        self.commit()
        return points
    
    def getSample(self, dbname, numRows, offset=0):
        cursor = self.cursor()
        samples = []
        try:
            cursor.execute('SELECT vector_hash, array_agg(asset_id) FROM %s GROUP BY vector_hash ORDER BY vector_hash LIMIT %d OFFSET %d'%(dbname,numRows,offset))
            samples = [{'id': str(row[0]), 'assets': row[1]} for row in cursor]
        except (Exception, psycopg2.Error) as error:
            print("Failed to lookup sample: %s"%error)       
        cursor.close()
        self.commit()
        return samples
        


class AssetDatabaseError(Exception):
    def __init__(self, message, suberror=None):
        self.message = None
        self.suberror = None
    def __str__(self):
        if self.suberror:
            return 'AssetDatabaseError: %s. Suberror: %s. '%(self.message, self.suberror)
        else:
            return 'AssetDatabaseError: %s. '%(self.message)