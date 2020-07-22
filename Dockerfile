FROM python:3.7-buster

WORKDIR /code

#ADD conf/pip.conf /root/.pip/

# Setup
COPY requirements.txt /code/requirements.txt
RUN 	pip install --upgrade numpy && \
	pip install -r requirements.txt

COPY PyCharmProject/src/app .

#Make python print immediately
ENV PYTHONUNBUFFERED 1

CMD gunicorn --bind 0.0.0.0:5000 wsgi
