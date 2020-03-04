from fastavro import reader, parse_schema
import requests
import json
import numpy as np


URL = 'http://0.0.0.0:5001/insert'

DIMENSIONS = 2048

DB_NAME = 'rtest2'

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
max_vectors = 1000000

with open(vectors_file, 'rb') as fo:
    c = 0
    for record in reader(fo):
        record_json = json.dumps(record)
        vector_list = record['vector']
        vector = np.asarray(vector_list)
        norm_vector = vector / np.linalg.norm(vector)
        norm_vector_list = norm_vector.tolist()

        payload = {'name': record['name'], 'vector': norm_vector_list}
        r = requests.put(url=URL, json=payload, params={'dbname': DB_NAME})

        c = c + 1
        if c >= max_vectors:
            break
