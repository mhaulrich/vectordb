FROM python:3.7-alpine
WORKDIR /code
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
RUN apk add --no-cache gcc g++ musl-dev linux-headers postgresql-dev 
# Setup 
# RUN pip install --upgrade setuptools
COPY requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt
COPY PyCharmProject/src/app .
CMD ["flask", "run"]
