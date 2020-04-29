FROM python:3.7-alpine
WORKDIR /code
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
RUN apk add --no-cache gcc g++ musl-dev linux-headers postgresql-dev 
# Setup 
# These dependencies are used in something in requirements
# They are installed here in order to create a docker layer that includes
# then because it kept getting reinstalled and it took forever.
RUN pip install --upgrade grpcio-tools
RUN pip install --upgrade numpy
COPY requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt
COPY PyCharmProject/src/app .

#Make python print immediately
ENV PYTHONUNBUFFERED 1

CMD ["flask", "run"]
#CMD gunicorn --bind 0.0.0.0:5000 wsgi
