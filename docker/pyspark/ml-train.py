import datetime
import argparse
import os
import glob
import tempfile

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    StringType,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

begin_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_path", type=str, help="Path to directory containing csv dataset."
)
parser.add_argument("output_dir", type=str, help="Path to output MLflow run_id.")
parser.add_argument("--tree_max_depth", type=int, default=5, help="Maximum tree depth hyperparameter to use for training a Decision Tree classifier.")
args = parser.parse_args()
print(vars(args))

spark = SparkSession.builder.getOrCreate()

schema = StructType(
    [
        StructField("mean_amount", DoubleType(), False),
        StructField("mean_oldbalanceOrg", DoubleType(), False),
        StructField("mean_newbalanceOrig", DoubleType(), False),
        StructField("mean_oldbalanceDest", DoubleType(), False),
        StructField("mean_newbalanceDest", DoubleType(), False),
        StructField("sum_isFraud", DoubleType(), False),
        StructField("nameOrig", DoubleType(), False),
        StructField("nameDest", StringType(), False),
    ]
)

file_list = [
    "file://" + f
    for f in glob.glob(os.path.join(args.dataset_path, "**/*.csv"), recursive=True)
]
print("file_list:")
print(file_list)

df = spark.read.load(file_list, format="csv", schema=schema, header=True)
df = df.withColumn(
    "orgDiff", df.mean_newbalanceOrig - df.mean_oldbalanceOrg
).withColumn("destDiff", df.mean_newbalanceDest - df.mean_oldbalanceDest)


(train, test) = df.randomSplit([0.8, 0.2], seed=12345)
train.cache()
test.cache()

va = VectorAssembler(
    inputCols=[
        "mean_amount",
        "mean_oldbalanceOrg",
        "mean_newbalanceOrig",
        "mean_oldbalanceDest",
        "mean_newbalanceDest",
        "orgDiff",
        "destDiff",
    ],
    outputCol="features",
)
clf = DecisionTreeClassifier(
    labelCol="sum_isFraud", featuresCol="features", seed=54321, maxDepth=args.tree_max_depth
)
pipeline = Pipeline(stages=[va, clf])
dt_model = pipeline.fit(train)

# Use BinaryClassificationEvaluator to evaluate our model
evaluatorPR = BinaryClassificationEvaluator(
    labelCol="sum_isFraud", rawPredictionCol="prediction", metricName="areaUnderPR"
)
evaluatorAUC = BinaryClassificationEvaluator(
    labelCol="sum_isFraud", rawPredictionCol="prediction", metricName="areaUnderROC"
)
# Build the best model (training and test datasets)
train_pred = dt_model.transform(train)
test_pred = dt_model.transform(test)
# Evaluate the model on training datasets
pr_train = evaluatorPR.evaluate(train_pred)
auc_train = evaluatorAUC.evaluate(train_pred)
# Evaluate the model on test datasets
pr_test = evaluatorPR.evaluate(test_pred)
auc_test = evaluatorAUC.evaluate(test_pred)

# Print out the PR and AUC values
print("PR train:", pr_train)
print("AUC train:", auc_train)
print("PR test:", pr_test)
print("AUC test:", auc_test)

# Log run in MLflow
pach_pipeline_name = os.environ["PPS_PIPELINE_NAME"]
pach_job_id = os.environ["PACH_JOB_ID"]
mlflow.set_tracking_uri("http://mlflow-svc.mlflow.svc.cluster.local:5000")
exp_name = "Default"
mlflow.set_experiment(experiment_name=exp_name)
with mlflow.start_run(run_name="{}@{}".format(pach_pipeline_name, pach_job_id)) as run:
    # Log Parameters and metrics
    mlflow.log_param("balanced", "yes")
    mlflow.log_param("seed", 54321)
    mlflow.log_param("maxDepth", args.tree_max_depth)
    mlflow.log_metric("PR train", pr_train)
    mlflow.log_metric("AUC train", auc_train)
    mlflow.log_metric("PR test", pr_test)
    mlflow.log_metric("AUC test", auc_test)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Write out trained model to Pachyderm output repository
        mlflow.spark.save_model(dt_model, tmpdirname)
        # Push serialized model into MLflow
        mlflow.log_artifacts(tmpdirname)
    run_id = run.info.run_id
    with open(os.path.join(args.output_dir, run_id), 'w') as fp:
        pass

print("Completed in {}".format(datetime.datetime.now() - begin_time))
