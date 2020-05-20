#/bin/bash
export VECTORDB_WD=$(pwd)
docker-compose build
docker-compose -f docker-compose.yml -f docker-compose.production.yml up #-d

