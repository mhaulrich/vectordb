import numpy as np
import json

dims = 8

rand_vector = np.random.rand(dims)

vector = rand_vector.tolist()

o = {}
o['name'] = 'v1'
o['vector'] = vector

json_str = json.dumps(o)
print(json_str)
