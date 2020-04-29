import sys
from flask import Flask, abort, Response
import json
from flask import request
import cProfile as profile
from VectorUtils import hash_vector
from DB_operations import AssetDatabase
from Milvus_operations import VectorIndex, IndexType
import os

# If running outside docker-compose run with
# python3 app.py <SQL HOST> <SQL PORT>  <MILVUS_HOST> <MILVUS PORT>

assetDB_host = "db"
assetDB_port = "5432"
milvus_host = "milvus"
milvus_port ="19530"
if len(sys.argv) == 5:
    assetDB_host = sys.argv[1]
    assetDB_port = sys.argv[2]

    milvus_host = sys.argv[3]
    milvus_port = sys.argv[4]


# In outer section of code
pr = profile.Profile()
pr.disable()
app = Flask(__name__)

assetDB = AssetDatabase(user="postgres",
                        password="mysecretpassword",
                        host=assetDB_host,
                        port=assetDB_port,
                        database="postgres")

vectorIndex = VectorIndex(host=milvus_host, port=milvus_port)

DEBUG = os.environ.get('VECTORDB_DEBUG') == 'true'


def abort_missing_parameters(method, need_to_have_parameters, additional_comment=None):
    error_string = 'ERROR:  Missing parameters. Method "' + method + "' needs the following parameters: " + ','.join(need_to_have_parameters) + '\n'
    if additional_comment:
        error_string = error_string + '\t' + additional_comment + '\n'
    abort(Response(error_string))


def abort_wrong_dimenions(dbname, reqqest_dims, db_dimensions):
    error_string = 'ERROR: vectors in ' + dbname + ' have ' + str(db_dimensions) + ' dimensions. Vector in request has ' + str(reqqest_dims) + '\n'
    abort(Response(error_string))


# Check if what is in meta corresponds with what vector_hash tables there are, and what there is in milvus
@app.route('/checkintegrity')
def check_db_integrety():
    
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
        return 'DB integrety bad. Check server output for details'


@app.route('/createdb')
def create_new_db():
    db_name = request.args.get('dbname')
    dimensions = int(request.args.get('dimensions'))
    index_type='IVFLAT'
    if assetDB.tableExists(db_name):
        return -1, 'Error: Vector database with name: ' + db_name + ' already exists'

    # Create in postgres
    assetDB.createVectorTable(db_name, dimensions, index_type)

    # Create in Milvus
    vectorIndex.createTable(db_name, dimensions, IndexType.IVFLAT)
    return "Created"


@app.route('/deletedb')
def delete_db():
    db_name = request.args.get('dbname')
    try:
        vectorIndex.deleteTable(db_name)
        assetDB.deleteVectorTable(db_name)
        return 'Table deleted'
    except Exception as error:
        print("Error while deleting table", error)
        return 'Problem with deleting table. See server log for details', 500


@app.route('/insert', methods=['PUT'])
def insert():
    pr.enable()
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    print(vector_with_id)
    if (db_name is None) or (vector_with_id is None):
        abort_missing_parameters('insert', ['dbname'], 'and vector in body')

    assets = vector_with_id['names']
    vectors = vectorIndex.make2DFloat(vector_with_id['vectors'])
    print(vectors)
    query_vector_dims = vectors.shape[1]
    db_dims = assetDB.getDimensions(db_name)
    if query_vector_dims != db_dims:
        abort_wrong_dimenions(db_name, query_vector_dims, db_dims)

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
        print('Adding: ' + str(len(index_vectors)) + ' to Index')

    if len(index_vectors) > 0:
        try:
            vectorIndex.insert(db_name, list(index_vectors.values()), list(index_vectors.keys()))
            assetDB.commit() #Finish the insert to assetDB
        except Exception as e:
            print("Something failed during insert: %s"%str(e))
            assetDB.rollback() #Roll back the insert to assetDB
            raise e

    pr.disable()
    return 'Successfully inserted %d vectors'%len(vectors)


@app.route('/lookupexact', methods=['PUT'])
def lookup_exact():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)
    assets_ids = assetDB.getAssets(db_name, vector_hash)
    assets_ids_json = json.dumps(assets_ids, indent=3)
    return assets_ids_json


@app.route('/lookup', methods=['PUT'])
def lookup():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)
    assets_ids = assetDB.getAssets(db_name, vector_hash)

    results = []

    # Find exact mataches
    # todo: Why this? Dosn't Milvus return 0 distance anyways?
    exact_matches = {'distance': 0, 'asset_ids': assets_ids}
    results.append(exact_matches)

    index_results = vectorIndex.lookup(db_name, vector, 20)
    for neighbour in index_results[0]:
        vector_hash = neighbour['id']
        distance = neighbour['distance']
        asset_ids = assetDB.getAssets(db_name, vector_hash)
        this_result = {'distance': distance, 'asset_ids': asset_ids}
        results.append(this_result)

    results_json = json.dumps(results, indent=3)
    return results_json


@app.route('/sampledb', methods=['GET'])
def sampledb():
    """Get a sample of 'n' rows from 'dbname'"""
    db_name = request.args.get('dbname')
    count = int(request.args.get('n', 50))
    rows = assetDB.getSample(db_name, count)
    if DEBUG:
        print("%-22s %s"%(("Hash","Asset")))
        for row in rows:
            print("%-22s %s"%row)
    return json.dumps(rows, indent=3)

@app.route('/listdbs', methods=['GET'])
def listdbs():
    """Describe the existing tables"""
    tables = assetDB.getExistingTables()
    asset_counts = assetDB.getNumberOfAssets(tables)
    table_infos = vectorIndex.describe(tables)
    for table_info, numAssets in zip(table_infos, asset_counts):
        table_info['no_assets'] = numAssets
    return json.dumps(table_infos, indent=3)


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


# init_db()

# if __name__ == "__main__":
#     app.run(port=5001)
