from fastavro import reader, parse_schema
import requests
import json
import numpy as np
from random import seed
from random import randint


URL = 'http://0.0.0.0:5001/lookup'

TEST_SET_SIZE = 20

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

vectors = []

# Reading
max_vectors = 10000000

with open(vectors_file, 'rb') as fo:
    c = 0
    for record in reader(fo):
        vector_list = record['vector']
        vector = np.asarray(vector_list)
        norm_vector = vector / np.linalg.norm(vector)
        vectors.append({'name': record['name'], 'vector' : norm_vector})

        c = c + 1
        if c >= max_vectors:
            break


random_indexes = []
seed(42)
for i in range(TEST_SET_SIZE):
    index = randint(0,len(vectors))
    random_indexes.append(index)

print(random_indexes)
out = []
for my_index in random_indexes:
    my_vector = vectors[my_index]['vector']
    my_name = vectors[my_index]['name']
    my_distances = []
    for v in vectors:
        distance = np.inner(my_vector, v['vector'])
        my_distances.append((v['name'], distance))
    my_distances.sort(key=lambda tup: tup[1], reverse=True)

    best_n_distances = []
    for i in range(20):
        best_n_distances.append(my_distances[i][0])
    result_record = {'name': my_name, 'vector': my_vector.tolist(), 'nbest': best_n_distances}
    out.append(result_record)

with open("ref20.json","w+") as f:
    for rec in out:
        f.write(json.dumps(rec) + '\n')
