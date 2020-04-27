import hashlib
from array import array
import marshal
import numpy as np


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
        m.update(num_int.to_bytes(16, 'little', signed=True))

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

    #Milvus needs ids to be positive, but of type signed-int,
    hash = abs(int.from_bytes(md5_bytes[:8], 'little', signed=True))
    return hash


def hash_vector(vector):
    return hash_vector6(vector)
