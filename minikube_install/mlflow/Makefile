VERSION=0.1.0
IMAGE_BASE=winderresearch/pachyderm-mlflow

build:
	docker build . -t ${IMAGE_BASE}:${VERSION}

push:
	docker push ${IMAGE_BASE}:${VERSION}

pull:
	docker pull ${IMAGE_BASE}:${VERSION}