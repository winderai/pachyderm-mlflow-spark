FROM python:3.8-slim

RUN pip install mlflow==1.22.0 psycopg2-binary==2.9.2 boto3==1.20.24

ENTRYPOINT ["mlflow", "server"]