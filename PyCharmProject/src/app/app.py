from flask import Flask, abort, Response
import json
from flask import request
import cProfile as profile
from VectorUtils import hash_vector
from DB_operations import *
from Milvus_operations import *

# If running outside docker-compose run with
# python3 app.py <SQL HOST> <SQL PORT>  <MILVUS_HOST> <MILVUS PORT>

if len(sys.argv) == 5:
    postgres_host = sys.argv[1]
    postgres_port = sys.argv[2]
    set_postsres_host(postgres_host, postgres_port)

    milvus_host = sys.argv[3]
    milvus_port = sys.argv[4]
    set_milvus_host(milvus_host, milvus_port)


# In outer section of code
pr = profile.Profile()
pr.disable()
app = Flask(__name__)


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
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM " + METATABLE_NAME)
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


@app.route('/createdb')
def create_new_db():
    db_name = request.args.get('dbname')
    dimensions = int(request.args.get('dimensions'))
    if check_table_exists(db_name):
        return -1, 'Error: Vector database with name: ' + db_name + ' already exists'

    # Create in postgres
    create_vector_table_in_db(db_name, dimensions)

    # Create in Milvus
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
        cursor.execute('DELETE FROM ' + METATABLE_NAME + ' WHERE name = %s', (db_name,))
    except (Exception, psycopg2.Error) as error:
        all_ok = False
        print("Error while trying to delete table in metatable.", error)
    cursor.close()

    conn.commit()

    if all_ok:
        return 'Table deleted'
    else:
        return 'Problem with deleting table. See server log for details'


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


@app.route('/lookupexact', methods=['PUT'])
def lookup_exact():
    db_name = request.args.get('dbname')
    vector_with_id = request.json
    vector = vector_with_id['vector']
    vector_hash = hash_vector(vector)
    assets_ids = get_assets(db_name, vector_hash)
    assets_ids_json = json.dumps(assets_ids)
    return assets_ids_json


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

    cursor.execute("SELECT name FROM " + METATABLE_NAME)
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
        # index_file_size = milvus_table.index_file_size
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
