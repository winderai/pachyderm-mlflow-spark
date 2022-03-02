import datetime
import argparse
import glob
import os

import pandas as pd

begin_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument(
    "raw_dataset_path", type=str, help="Path to directory containing raw csv datasets."
)
parser.add_argument("output_dir", type=str, help="Path to output csv chunks.")
parser.add_argument(
    "--subsample", help="increase output verbosity", action="store_true"
)
args = parser.parse_args()

for f in glob.glob(os.path.join(args.raw_dataset_path, "*.csv")):
    print("Processing csv dataset: {}".format(f))
    df = pd.read_csv(f)
    # Keep only transaction types that include any fraud activity
    df = df[(df["type"] == "PAYMENT") | (df["type"] == "CASH_OUT")]
    # Originally "nameOrig" is string starting with a letter; remove that and transform it to integer.
    # This will make grouping-by "nameOrig" much faster
    df["nameOrig"] = df["nameOrig"].apply(lambda x: x[1:]).astype(int)
    if args.subsample:
        n = 25
        df = pd.concat(
            [
                df[df["isFraud"] == 0].sample(n, random_state=42),
                df[df["isFraud"] == 1].sample(n, random_state=42),
            ]
        ).sample(frac=1.0)
    else:
        df = df.sample(frac=1.0)
    df.groupby("nameOrig").apply(
        lambda x: x.to_csv(
            os.path.join(args.output_dir, "{}.csv".format(x["nameOrig"].values[0]))
        )
    )
print("Completed in {}".format(datetime.datetime.now() - begin_time))
