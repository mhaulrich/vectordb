#/bin/bash
docker-compose build && docker-compose run web python web_test.py
