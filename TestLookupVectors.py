from fastavro import reader, parse_schema
import requests
import json

URL = 'http://0.0.0.0:5001/lookup'


DB_NAME = 'rtest2'

# path to vectors
ref_vector_file = 'ref20.json'

ref_vectors = []

with open(ref_vector_file, 'r') as f:
    for line in f.readlines():
        rec = json.loads(line)
        ref_vectors.append(rec)


ave_rec_sum = 0
for record in ref_vectors:
    lookup_vector = record['vector']
    payload = {'vector': lookup_vector}
    r = requests.put(url=URL, json=payload, params={'dbname': DB_NAME})
    results = r.json()
    retrieved_ids = {}
    for result in results:
        for id in result['asset_ids']:
            retrieved_ids[id] = True
    found = 0
    for ref_name in record['nbest']:
        if ref_name in retrieved_ids:
            found = found + 1
    rec = found / len(record['nbest'])
    ave_rec_sum = ave_rec_sum + rec

ave_rec = ave_rec_sum / len(ref_vectors)
print('Averaged recall:', "%.4f" % ave_rec)