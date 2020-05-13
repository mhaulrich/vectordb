# VectorDB

VectorDB is an extension to the Milvus Similariy Search Engine, allowing string-based assets to be associated with feature vectors. This is for example convenient in cases where you want to keep track of what files generated what feature vectors in an efficient and persistent manor. VectorDB enables you to insert a feature vector and an associated asset, e.g. the filename. When you later execute a Nearest Neighbour (Similarity) search on a query point, VectorDB will return the nearest points as well as their associated assets - which may often be more than one asset in cases where different documents generated the same feature vectors.  
VectorDB also tries to group identical feature vectors prior to insertion in the vector index, thus avoiding duplicate entries. In scenarios where same point is added multiple times, e.g. because the same data is processed twice, this is convenient. 

VectorDB runs on docker-compose. The vector indexing uses the Milvus Similarity Search Engine, and the asset store uses a Postgres database. The rest-interface and logic binding the two databases together is implemented in Python using Flask for the webpart. 

## How to run

Use the `up.sh` shell script. This creates an environment variable with your current directory, builds the docker-compose project and runs it.

## UnitTest the project

After having started vectordb (as described above), execute:
```
docker-compose run web python web_test.py
```

## Usage
Vectordb exposes a REST interface. Examples of how to communicate with the interface is presented below.

### Databases

#### List all databases:
Get a list of all existing databases and their properties:
```
curl localhost:5000/databases
```
#### Show a specific database:
Get the properties of a specific database. Below, the database `test` is inspected:
```
curl localhost:5000/databases/test
```
#### Create a new database:
Create a new database where the database name (must be a string) is specified by the parameter `dbname` and the dimensionality of the database (i.e. vectors inserted into it) is specified by the parameter `dimensions`:
```
curl localhost:5000/databases -F dbname=test -F dimensions=3
```
#### Delete a database:
Deletion is done via a HTTP DELETE request to the resource. Below, the database `test` is deleted:
```
curl localhost:5000/databases/test -X DELETE
```

### Points
#### List points in a database:
Get a sample list of the points in a database. Optionally number of points and a start offset can be specified by the parameters `count` and `offset` respectively:
```
curl 'localhost:5000/databases/test/points'
curl 'localhost:5000/databases/test/points?count=5&offset=0'
```
#### Show a specific point in a DB:
Get the coordinates and assets of a specific point in a database. Below, the point with ID '6387894434316547108' from table `test` is retrieved:
```
curl localhost:5000/databases/test/points/6387894434316547108/
```

#### Insert vector(s) into a database
Insert a list of vectors and their corresponding assets into a database. The field `vectors` must be a 2D list of numbers and the field `assets` must be a 1D list of string-assets. The length of `vectors` must match the length of `assets`. Below a single vector with coordinates `[1,2,3]` and asset `myAsset` is inserted into the `test` database.
```
curl localhost:5000/databases/test/ -H 'Content-Type: application/json' -X POST -d '{"assets": ["myAsset"], "vectors": [[1,2,3]]}'
```

#### Exact search for vector
Search for all neighbours that have exactly `0` distance to the query vectors. This corresponds to getting all assets associated with the query vector. Below an exact search is executed for the query vector `[1,2,3]`
```
curl localhost:5000/databases/test/lookup/ -H 'Content-Type: application/json' -X POST -d '{"exact": true, "vectors": [[1,2,3]]}'
```
#### Nearest Neighbour search for vector
Search for the `count` nearest neighbours to the query vectors specified in `vectors`. Below the 5 nearest neighbours are requested for the query vector `[1,2,3]`
```
curl localhost:5000/databases/test/lookup/ -H 'Content-Type: application/json' -X POST -d '{"exact": false, "count": 5, "vectors": [[1,2,3]]}'
```
### Other
#### Check db integrity:
Checks the database integrity, verifying that corresponding vectorindexes and assetstores exists.
```
curl localhost:5000/check
```
