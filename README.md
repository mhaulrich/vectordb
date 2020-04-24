# vectordb

## How to run
Change working dir in .env, build, and run
```
export VECTORDB_WD=$(pwd)
docker-compose build
docker-compose up
```

## Example commands

Check db integrity (will also list created DBs):
```
curl 0.0.0.0:5000/checkintegrity
```
Create a new DB with name and dimenions
```
curl "0.0.0.0:5000/createdb?dbname=test&dimensions=8"
```
Delete DB
```
curl 0.0.0.0:5000/deletedb?dbname=test
```
List databases
```
curl 0.0.0.0:5000/listdbs | jq

```
Insert vector into DB
```
curl -H 'Content-Type: application/json' -X PUT -d '{"name": "v1", "vector": [0.00793861557915232, 0.9997302132320715, 0.37542373014507036, 0.6712801110144234, 0.35517993228837497, 0.4425925168063203, 0.48268047072393094, 0.15620825516058134]}' "0.0.0.0:5000/insert?dbname=test"
```
Exact search for vector
```
curl -H 'Content-Type: application/json' -X PUT -d '{"vector": [0.00793861557915232, 0.9997302132320715, 0.37542373014507036, 0.6712801110144234, 0.35517993228837497, 0.4425925168063203, 0.48268047072393094, 0.15620825516058134]}' "0.0.0.0:5000/lookupexact?dbname=test"
```
NN (and exact) search for vector
```
curl -H 'Content-Type: application/json' -X PUT -d '{"vector": [0.00793861557915232, 0.9997302132320715, 0.37542373014507036, 0.6712801110144234, 0.35517993228837497, 0.4425925168063203, 0.48268047072393094, 0.15620825516058134]}' "0.0.0.0:5000/lookup?dbname=test"
```

