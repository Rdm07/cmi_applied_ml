FROM python:3.9-bullseye

COPY requirements.txt ./

RUN pip install -r requirements.txt

ENV BASE_DIR="/"

RUN mkdir ${BASE_DIR}/app_data
RUN mkdir /usr/local/nltk_data

COPY ./data ${BASE_DIR}/app_data/data
COPY ./models ${BASE_DIR}/app_data/models
COPY ./src ${BASE_DIR}/app_data/src
COPY ./test ${BASE_DIR}/app_data/test
COPY ./nltk_data /usr/local/nltk_data/


WORKDIR	${BASE_DIR}/app_data

ENTRYPOINT python ${BASE_DIR}/app_data/src/app.py