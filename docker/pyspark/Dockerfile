FROM python:3.8-slim
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget curl unzip less && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda install -y -c conda-forge openjdk==11.0.9.1
RUN pip install pyspark==3.2.0 numpy==1.21.4 mlflow==1.22.0 boto3==1.20.24

ENV SPARK_LOCAL_IP='127.0.0.1'
ENV GIT_PYTHON_REFRESH=quiet
ENV AWS_ACCESS_KEY_ID=1
ENV AWS_SECRET_ACCESS_KEY=1
ENV MLFLOW_S3_ENDPOINT_URL=http://pachd.pachyderm.svc.cluster.local:30600

COPY etl.py /root/etl.py
COPY ml-train.py /root/ml-train.py