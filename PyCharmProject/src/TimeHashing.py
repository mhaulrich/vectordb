from fastavro import reader, parse_schema
import requests
import json
import numpy as np
import hashlib
from timeit import default_timer as timer
from array import array
import marshal
import time


def hash_vector1(vector):
    m = hashlib.md5()
    for num in vector:
        num_str = '%.5f' % num
        m.update(bytes(num_str, encoding='utf-8'))

    # vector_str = ','.join(['%.5f' % num for num in vector])
    # m.update(bytes(vector_str, encoding='utf-8'))
    md5_bytes = m.digest()
    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


def hash_vector2(vector):
    vector_str = ','.join(['%.5f' % num for num in vector])
    m = hashlib.md5()
    m.update(bytes(vector_str, encoding='utf-8'))
    for num in vector:
        num_str = '%.5f' % num
        m.update(bytes(num_str, encoding='utf-8'))

    # vector_str = ','.join(['%.5f' % num for num in vector])
    # m.update(bytes(vector_str, encoding='utf-8'))
    md5_bytes = m.digest()
    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


def hash_vector3(vector):
    m = hashlib.md5()
    for num in vector:
        num_int = int(num * 100000)
        m.update(num_int.to_bytes(16, 'little', signed=False))

    # vector_str = ','.join(['%.5f' % num for num in vector])
    # m.update(bytes(vector_str, encoding='utf-8'))
    md5_bytes = m.digest()
    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


def hash_vector4(vector):
    int_list = [None] * len(vector)
    for i in range(len(vector)):
        int_list[i] = int(vector[i] * 100000)

    m = marshal.dumps(int_list)
    md5_bytes = hashlib.md5(m).digest()

    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


def hash_vector5(vector):
    int_list = [None] * len(vector)
    for i in range(len(vector)):
        int_list[i] = int(vector[i] * 100000)

    a = array('i', int_list)
    m = marshal.dumps(a)
    md5_bytes = hashlib.md5(m).digest()

    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


def hash_vector6(vector):
    nparray = np.array(vector) * 100000
    nparray_int = nparray.astype('i2')
    m = marshal.dumps(nparray_int)
    md5_bytes = hashlib.md5(m).digest()

    hash = int.from_bytes(md5_bytes[:8], 'little', signed=True)
    return hash


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
