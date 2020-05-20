import sys
import logging
from flask import Flask, abort, Response
from flask_restful import reqparse, abort, Api, Resource
import json
from flask import request
import cProfile as profile
from VectorUtils import hash_vector
from AssetDatabase import AssetDatabase
from VectorIndex import VectorIndex, IndexType
import os
import traceback

# In outer section of code
pr = profile.Profile()
pr.disable()
app = Flask(__name__)
app.url_map.strict_slashes = False
api = Api(app)

assetDB_host = "db"
assetDB_port = "5432"
milvus_host = "milvus"
milvus_port ="19530"


if __name__ == "__main__":
    # If running outside docker-compose run with
    # python3 app.py <SQL HOST> <SQL PORT>  <MILVUS_HOST> <MILVUS PORT>
    if len(sys.argv) == 5:
        assetDB_host = sys.argv[1]
        assetDB_port = sys.argv[2]
    
        milvus_host = sys.argv[3]
        milvus_port = sys.argv[4]
    app.run(port=5001)
else: 
    #Hook up gunicorn logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)



assetDB = AssetDatabase(user="postgres",
                        password="mysecretpassword",
                        host=assetDB_host,
                        port=assetDB_port,
                        database="postgres")

vectorIndex = VectorIndex(host=milvus_host, port=milvus_port)

DEBUG = os.environ.get('VECTORDB_DEBUG') == 'true'


def check_abort_wrong_dimenions(dbname, request_dims):
    db_dims = assetDB.getDimensions(dbname)
    if request_dims != db_dims:
        abort(Response('ERROR: vectors in %s have %d dimensions. Vector in request has %d.'%(dbname, request_dims, db_dims), 400))
        
def check_abort_missing_db(dbname):
    if not assetDB.tableExists(dbname):
        abort(Response("ERROR: Table '%s' does not exist."%(dbname), 404))



#Arg parsers
parser_newDB = reqparse.RequestParser()
parser_newDB.add_argument('dbname', type=str, required=True, help='Name of the database (str)')
parser_newDB.add_argument('dimensions', type=int, required=True, help='Dimensionality of the database (int)')

# testparser.add_argument('list', type=float, required=True, action='append', help='test list input')
# testparser.add_argument('array', type=list, required=True, action='append', help='test 2d array input')
# testparser.add_argument('stringlist', type=str, required=True, action='append', help='test string list input')
        

parser_newPoint = reqparse.RequestParser()
parser_newPoint.add_argument('vectors', type=list, action='append', required=True, help='List of vectors to insert in database')
parser_newPoint.add_argument('assets', type=str, action='append', required=True, help='List of associated assets to insert in database')

parser_listPoints = reqparse.RequestParser()
parser_listPoints.add_argument('count', default=10, type=int, required=False, help='Number of points to return')
parser_listPoints.add_argument('offset', default=0, type=int, required=False, help='Offset in point query')

parser_lookup = reqparse.RequestParser()
parser_lookup.add_argument('vectors', type=list, action='append', required=True, help='List of vectors to lookup in database')
parser_lookup.add_argument('count', default=10, type=int, required=False, help='Number of nearest neighbours to return')
parser_lookup.add_argument('exact', default=False, type=bool, required=False, help='Only search for exact matches (distance=0)')


class DatabaseList(Resource):
    """Rest interface for listing and creating databases"""

    def get(self):
        """List all existing databases
        - curl http://localhost:5000/databases/ """
        #
        tables = assetDB.getExistingTables()
        asset_counts = assetDB.getNumberOfAssets(tables)
        table_infos = vectorIndex.describeTables(tables)
        for table_info, numAssets in zip(table_infos, asset_counts):
            table_info['no_assets'] = numAssets
        return table_infos
        
    
    def post(self):
        """Create a new database"""
        args = parser_newDB.parse_args()
        db_name = args['dbname']
        dimensions = args['dimensions']
        index_type='IVFLAT'
        if assetDB.tableExists(db_name):
            return "Error: Vector database with name '%s' already exists"%db_name, 409
        # Create in postgres
        assetDB.createVectorTable(db_name, dimensions, index_type)
        # Create in Milvus
        try:
            vectorIndex.createTable(db_name, dimensions, IndexType.IVFLAT)
        except Exception as e:
            print("Something failed during table creation: %s"%str(e))
            assetDB.rollback() #Roll back the insert to assetDB
            raise e
        assetDB.commit()
        # Get a description back
        table_info = vectorIndex.describeTables(db_name)
        table_info["no_assets"] = 0
        return table_info, 201

    

class Database(Resource):
    """Rest interface for individual databases"""
    
    def get(self, db_name):
        """Return information on a specific database
        - curl http://localhost:5000/databases/test """
        check_abort_missing_db(db_name)
        table_info = vectorIndex.describeTables(db_name)
        table_info['no_assets'] = assetDB.getNumberOfAssets(db_name)
        return table_info
    
    def delete(self, db_name):
        """Delete a database
        - curl http://localhost:5000/databases/test -X DELETE """
        if not vectorIndex.tableExists(db_name) and not assetDB.tableExists(db_name):
            abort(Response("ERROR: Table '%s' does not exist in either asset database or vector index."%(db_name), 404))
        errMsgs = []
        try:
            vectorIndex.deleteTable(db_name)
        except Exception as e:
            errMsgs.append(str(e))
        try:
            assetDB.deleteVectorTable(db_name)
        except Exception as e:
            errMsgs.append(str(e))
        if len(errMsgs) > 0:
            print(errMsgs)
            return 'Warning: the following errors occurred during delete: '+' | '.join(errMsgs), 200
        return '', 204
    
    def post(self, db_name):
        """Insert points and associated assets in the database
            There are three cases:
            1. Vector hash and asset already exists in db
                Nothing should be done
            2. Vector hash exists but this asset does not
                asset should be inserted into asset db.
                No vector should be added to vector index as it is already there
            3. New vector hash.
                (Vector hash, asset) should be inserted into db
                Vector should be inserted into vector index"""      
        pr.enable()
        args = parser_newPoint.parse_args()
        if DEBUG:
            print("Incoming args:\n%s"%args)
        vectors = vectorIndex.make2DFloat(args['vectors'])
        assets = args['assets']
        if DEBUG:
            print("Incoming vectors:\n%s"%vectors)
            print("Incoming assets:\n%s"%assets)
        if len(vectors) != len(assets):
            abort(Response("ERROR: length of vectors must match length of assets. %d != %d"%(len(vectors),len(assets)), 400))
        query_vector_dims = vectors.shape[1]
        check_abort_wrong_dimenions(db_name, query_vector_dims)
        
        vector_hashes = []
        for vector in vectors:
            vector_hashes.append(hash_vector(vector))
    
        # Find hashes and vectors that should be added to milvus
        # We do this in a dict to avoid actually adding the same vector more than once
        vector_hash_exists = assetDB.insertVectorHashes(db_name, vector_hashes, assets)
        index_vectors = {}
        for i, exists in enumerate(vector_hash_exists):
            if not exists:
                index_vectors[vector_hashes[i]] = vectors[i]
        if DEBUG:
            print('Adding: %d/%d to Index'%(len(index_vectors),len(vectors)))
    
        if len(index_vectors) > 0:
            try:
                vectorIndex.insert(db_name, list(index_vectors.values()), list(index_vectors.keys()))
            except Exception as e:
                print("Something failed during insert: %s"%str(e))
                assetDB.rollback() #Roll back the insert to assetDB
                raise e
            assetDB.commit() #Finish the insert to assetDB
        pr.disable()
        return vector_hashes, 201
    

class PointList(Resource):
    """Rest interface for points"""

    def get(self, db_name):
        """Get a sample of 'count' rows from 'dbname' starting a 'offset'"""
        args = parser_listPoints.parse_args()
        samples = assetDB.getSample(db_name, args['count'], args['offset'])
        if DEBUG:
            print("%-22s %s"%(("Hash","Asset")))
            for sample in samples:
                first = True
                for asset in sample['assets']:    
                    if first:
                        print("%-22s %s"%(sample['id'],asset))
                        first = False
                    else:
                        print("%-22s %s"%('',asset))
                        
        return samples
    
class Point(Resource):
    """Rest interface for individual points"""

    def get(self, db_name, point_hash):
        """Get point and assets for a provided point_hash"""
        return {
            'id': int(point_hash),
            'vector': vectorIndex.describePoint(db_name, int(point_hash)),
            'assets': assetDB.getAssets(db_name, point_hash)
            }
    #TODO:
    # def delete(self, db_name, point_hash):
    #     "", 204
    
class Lookup(Resource):
    """Rest interface for nearest neighbour searches"""

    def post(self, db_name):
        """Lookup the nearest neighbours to the vectors provided."""
        args = parser_lookup.parse_args()
        if DEBUG:
            print("Incoming args:\n%s"%args)
        count = args['count']
        vectors = vectorIndex.make2DFloat(args['vectors'])
        exact = args['exact']
        hashes = [hash_vector(vector) for vector in vectors]
        
        results = []
        if exact: # Find exact mataches
            for vector_hash in hashes:
                assets = assetDB.getAssets(db_name, vector_hash)
                vectorResult = {
                    'neighbours': [{'id': vector_hash, 'distance': 0.0, 'assets': assets}],
                    'queryid': vector_hash
                    }     
                results.append(vectorResult)
        else: #NN lookup
            index_results = vectorIndex.lookup(db_name, vectors, count)
            for index_result, vector_hash in zip(index_results, hashes):
                print("Got %d neighbours for hash %d"%(len(index_result),vector_hash))
                for neighbour in index_result: 
                    neighbour['assets'] = assetDB.getAssets(db_name, neighbour['id'])
                    # if neighbour['id'] == vector_hash: #Exactly the same - override distance
                    #     neighbour['distance'] = 0.0
                vectorResult = {
                    'neighbours': index_result, 
                    'queryid': vector_hash
                    }     
                results.append(vectorResult)

        return results
    
class Flush(Resource):
    """Rest interface for forcing flushing of VectorIndex data to disk.
    This is done automatically every second"""

    def post(self, db_name):
        """Flush VectorIndex data to disk"""
        if DEBUG:
            print("Flushing VectorIndex to disk...")
        vectorIndex.flushTable(db_name)
    
class Check(Resource):
    """Rest interface for checking database integrety"""
    
    def get(self):
        """Check if what is in meta corresponds with what vector_hash tables 
        there are, and what there is in vector index"""
        tableNames = assetDB.getExistingTables()
        print("Number of tables in meta: %d"%len(tableNames))
    
        all_good = True
        print('Table name\tExists in postgres\tExists in Milvus')
        for tableName in tableNames:
            exists_in_postgres = assetDB.tableExists(tableName)
            exists_in_milvus = vectorIndex.tableExists(tableName)
            print(tableName + '\t\t' + str(exists_in_postgres) + '\t\t\t\t' + str(exists_in_milvus))
            all_good = all_good and exists_in_postgres and exists_in_milvus
        if all_good:
            return 'All ok'
        else:
            return 'Database integrety bad. Check server output for details', 500
        
        
class Shutdown(Resource):
    """Rest interface for shutting down database"""
    
    def post(self):
        """Check if what is in meta corresponds with what vector_hash tables 
        there are, and what there is in vector index"""
        pr.dump_stats('profile.pstat')
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return 'Server shutting down...'
    
class Test(Resource):
    """Rest interface for shutting down database"""
    
    def post(self):
        import numpy as np
        testparser = reqparse.RequestParser(bundle_errors=False, )
        testparser.add_argument('list', type=float, required=True, action='append', help='test list input')
        testparser.add_argument('array', type=list, required=True, action='append', help='test 2d array input')
        testparser.add_argument('stringlist', type=str, required=True, action='append', help='test string list input')
        # testparser.add_argument('number', type=float, required=True, help='test float input')
        # testparser.add_argument('array', type=np.array, required=False, help='test numpy array input')
        args = testparser.parse_args()
        
        print(args)
        return args 
        

    

api.add_resource(DatabaseList, '/databases/')
api.add_resource(Database, '/databases/<db_name>/')
api.add_resource(PointList, '/databases/<db_name>/points/')
api.add_resource(Point, '/databases/<db_name>/points/<point_hash>/')
api.add_resource(Lookup, '/databases/<db_name>/lookup/')
api.add_resource(Flush, '/databases/<db_name>/flush/')
api.add_resource(Check, '/check/')
api.add_resource(Shutdown, '/shutdown/')
api.add_resource(Test, '/test/')

