from fastavro import reader, parse_schema
import json
from app.VectorUtils import *
from timeit import default_timer as timer





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
max_vectors = 50000


vectors = []

with open(vectors_file, 'rb') as fo:
    c = 0

    for record in reader(fo):
        record_json = json.dumps(record)
        vector_list = record['vector']
        vector = np.asarray(vector_list)
        norm_vector = vector / np.linalg.norm(vector)
        norm_vector_list = norm_vector.tolist()

        vectors.append(norm_vector_list)

        c = c + 1
        if c >= max_vectors:
            break

print(len(vectors))

start = timer()
for v in vectors:
    hash_vector1(v)
end = timer()
t = end - start
print('Method 1: ', t)


start = timer()
for v in vectors:
    hash_vector2(v)
end = timer()
t = end - start
print('Method 2: ', t)


start = timer()
for v in vectors:
    hash_vector3(v)
end = timer()
t = end - start
print('Method 3: ', t)


start = timer()
for v in vectors:
    hash_vector4(v)
end = timer()
t = end - start
print('Method 4: ', t)


start = timer()
for v in vectors:
    hash_vector5(v)
end = timer()
t = end - start
print('Method 5: ', t)


start = timer()
for v in vectors:
    hash_vector6(v)
end = timer()
t = end - start
print('Method 6: ', t)
