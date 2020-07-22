#/bin/bash
set -e
export VECTORDB_WD=$(pwd)
docker-compose build
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up #-d
