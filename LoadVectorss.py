from fastavro import reader, parse_schema
import requests
import json
import numpy as np


URL = 'http://0.0.0.0:5001/insertvectors'
SHUTDOWNURL = 'http://0.0.0.0:5001/shutdown'

DIMENSIONS = 2048

BATCH_SIZE = 20

DB_NAME = 'rtest14'

# path to vectors
vectors_file = '/stuff/imagenet/imagenetvectors-120000.avro'

schema = {
    'type': 'record',
    'name': 'VectorWithName',
    'namespace': 'mwh.image',
    'fields': [{
        'name': 'name',
        'type': 'string'
    }, {
        'name': 'vector',
        'type': {
            'type': 'array',
            'items': 'float'
        }
    }]
}

parsed_schema = parse_schema(schema)

# Reading
max_vectors = 10000

batches_added = 0

with open(vectors_file, 'rb') as fo:
    c = 0
    batch = []
    for record in reader(fo):
        record_json = json.dumps(record)
        vector_list = record['vector']
        vector = np.asarray(vector_list)
        norm_vector = vector / np.linalg.norm(vector)
        norm_vector_list = norm_vector.tolist()

        payload = {'name': record['name'], 'vector': norm_vector_list}
        batch.append(payload)
        if len(batch) == BATCH_SIZE:
            r = requests.put(url=URL, json=batch, params={'dbname': DB_NAME})
            batch = []
            batches_added = batches_added + 1

        c = c + 1
        if c >= max_vectors:
            break
    # Add remaining
    if len(batch) > 0:
        r = requests.put(url=URL, json=batch, params={'dbname': DB_NAME})
        batches_added = batches_added + 1

print('Added ' + str(batches_added) + 'batches')
requests.get(url=SHUTDOWNURL)