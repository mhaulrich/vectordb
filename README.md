# vectordb

## How to run
Change working dir in .env, build, and run
```
export VECTORDB_WD=$(pwd)
docker-compose build
docker-compose up
```

## Example commands

Check db integrity (will also print created DBs on server):
```
curl localhost:5000/check
```

List all databases:
```
curl localhost:5000/databases
```
Create a new DB with name and dimenions:
```
curl localhost:5000/databases -F dbname=test -F dimensions=3
```
Delete DB:
```
curl localhost:5000/databases/test -X DELETE
```

List points in a DB:
```
curl 'localhost:5000/databases/test/points?count=5&offset=0'
```

Insert vector(s) into DB
```
curl localhost:5000/databases/test/ -H 'Content-Type: application/json' -X POST -d '{"assets": [["myAsset"]], "vectors": [[1,2,3]]}'
```

Exact search for vector
```
curl localhost:5000/databases/test/lookup/ -H 'Content-Type: application/json' -X POST -d '{"exact": true, "vectors": [[1,2,3]]}'
```
NN (and exact) search for vector
```
curl localhost:5000/databases/test/lookup/ -H 'Content-Type: application/json' -X POST -d '{"exact": false, "vectors": [[1,2,3]]}'
```

