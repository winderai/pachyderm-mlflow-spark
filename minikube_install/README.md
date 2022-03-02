# Minikube installation

## Prerequisites

1. Docker installed
1. Kubernetes v1.22.3
1. Kubectl v1.22.3
1. [`minikube`](https://minikube.sigs.k8s.io/docs/) v1.23.0
1. [`helm`](https://helm.sh/docs/intro/install/) v3.7.1
1. [`pachctl`](https://docs.pachyderm.com/latest/getting_started/local_installation/#install-pachctl) v2.0.1

## Launch Minikube

```
minikube start --cpus 4 --memory 8000 --kubernetes-version=v1.22.3
```

## Pachyderm Local Deploy

```
helm install pachd pach/pachyderm --set deployTarget=LOCAL --namespace pachyderm --create-namespace
```

Run `kubectl get pods --namespace=pachyderm` and wait until all pods are in `Running` status.

## Setup `pachctl`

To connect to your new pachyderm instance, run:
```
pachctl config import-kube --namespace pachyderm local --overwrite
pachctl config set active-context local
```

Check your Pachyderm install with `pachctl version`, it should return the following:
```
COMPONENT           VERSION
pachctl             2.0.1
pachd               2.0.1
```

## Deploy MLflow

Create a Pachyderm repository that MLflow will use for storing artifacts:
```
pachctl create repo artifacts
pachctl create branch artifacts@master
```

Create namespace and launch MLflow.
It'll pull the [winderresearch/pachyderm-mlflow](https://hub.docker.com/repository/docker/winderresearch/pachyderm-mlflow) image from Docker Hub.
```
kubectl apply -f mlflow/namespace.yaml
kubectl apply -f mlflow/deployment.yaml --namespace=mlflow
kubectl apply -f mlflow/service.yaml --namespace=mlflow
```

Verify resources have been crerated:
```
kubectl get deployments.apps --namespace=mlflow
kubectl get svc --namespace=mlflow
```

Run `kubectl get pods --namespace=mlflow` and wait until all pods are in `Running` status.

Launch port forwarding in a separate terminal:
```
kubectl port-forward --namespace=mlflow service/mlflow-svc 30001:5000
```

Verify you can access MLflow web UI at: http://127.0.0.1:30001/

You're all set up if `kubectl get pods --all-namespaces` returns the following pods:

```
NAMESPACE     NAME                                 READY   STATUS    RESTARTS      AGE
kube-system   coredns-78fcd69978-k5xn2             1/1     Running   0             20m
kube-system   etcd-minikube                        1/1     Running   0             20m
kube-system   kube-apiserver-minikube              1/1     Running   0             20m
kube-system   kube-controller-manager-minikube     1/1     Running   0             20m
kube-system   kube-proxy-jmx2d                     1/1     Running   0             20m
kube-system   kube-scheduler-minikube              1/1     Running   0             20m
kube-system   storage-provisioner                  1/1     Running   1 (19m ago)   20m
mlflow        mlflow-deployment-667565b879-mvljg   1/1     Running   0             75s
pachyderm     etcd-0                               1/1     Running   0             8m8s
pachyderm     pachd-7c48845bbf-sndxq               1/1     Running   0             8m8s
pachyderm     pg-bouncer-58f48b7b49-pllb5          1/1     Running   0             8m8s
pachyderm     postgres-0                           1/1     Running   0             8m8s
```

Congrats! You've installed all prerequistes and can now move on to the [demo notebook](https://gitlab.com/WinderAI/pachyderm/databricks-demo/-/blob/main/demo.ipynb).

## Cleanup

```
minikube delete
```

## Credits 

This project was created by [Winder.ai, an ML consultancy](https://winder.ai/), and funded by [Pachyderm](https://pachyderm.com).