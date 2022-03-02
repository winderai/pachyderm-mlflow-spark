build:
	make -C minikube_install/mlflow build
	make -C docker/pyspark build
	make -C docker/split-data build

pull:
	make -C minikube_install/mlflow pull
	make -C docker/pyspark pull
	make -C docker/split-data pull

push:
	make -C minikube_install/mlflow push
	make -C docker/pyspark push
	make -C docker/split-data push