FROM python:3.9-bullseye

COPY requirements.txt ./

RUN pip install -r requirements.txt

ENV BASE_DIR="."

RUN mkdir ${BASE_DIR}/app_data
RUN mkdir /usr/local/nltk_data

COPY /home/rohan/nltk_data/* /usr/local/nltk_data/


WORKDIR	${BASE_DIR}/app_data

# ENTRYPOINT python ${BASE_DIR}/app_data/src/app.py

CMD ["/bin/bash"]