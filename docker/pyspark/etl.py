import datetime
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, mean as _mean, sum as _sum, col

begin_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_path", type=str, help="Path to directory containing csv chunks."
)
parser.add_argument("output_dir", type=str, help="Path to output aggregated csv data.")
args = parser.parse_args()

# Create Spark session
spark = SparkSession.builder.getOrCreate()

df = spark.read.option("header", "true").csv(os.path.join(args.dataset_path, "*.csv"))

# Aggregate records
df_agg = df.select(
    _mean(col("amount")).alias("mean_amount"),
    _mean(col("oldbalanceOrg")).alias("mean_oldbalanceOrg"),
    _mean(col("newbalanceOrig")).alias("mean_newbalanceOrig"),
    _mean(col("oldbalanceDest")).alias("mean_oldbalanceDest"),
    _mean(col("newbalanceDest")).alias("mean_newbalanceDest"),
    _sum(col("isFraud")).alias("sum_isFraud"),
)

# Add column with fixed value
nameOrig = df.first()["nameOrig"]
nameDest = df.first()["nameDest"]
df_agg = df_agg.withColumn("nameOrig", lit(nameOrig))
df_agg = df_agg.withColumn("nameDest", lit(nameDest))

df_agg.write.csv(os.path.join(args.output_dir, "{}".format(nameOrig)), header=True)
print("Completed in {}".format(datetime.datetime.now() - begin_time))
