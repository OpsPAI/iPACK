import csv
import logging
import os
import pandas as pd
import argparse
import time
from ipack.parser.AlertDrain import AlertDrain
from ipack.utils import set_logger

# python alert_parsing.py --file "../data/train/historical_alerts.csv" --outdir "../data/train/" B
# python alert_parsing.py --file "../data/train/train_ticket_alert_pairs.csv" --outdir "../data/train/" B
# python alert_parsing.py --file "../data/test/test_ticket_alert_pairs.csv" --outdir "../data/test/" B

set_logger("ipack_results/execution.log")
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--outdir", type=str)
parser.add_argument("--content_column", type=str)
params = vars(parser.parse_args())
filename = params["file"]
content_column = "AlertTitle"

# Parameters for Drain
st = 0.6
depth = 3
regex = [
    "(?<=^|[\s\W])[0-9]+?(?=$|[\s\W])",
    "(^\[.+?\])+",
    "(\w*-\w*)+",
    "(\w*_\w*)+",
]
# partition_key = ["AlertOwningComponent", "AlertMonitorId"]
partition_key = []
outdir = params["outdir"]
output_filename = os.path.join(
    outdir,
    os.path.basename(filename).replace(".csv", "_parsed.csv"),
)
os.makedirs(outdir, exist_ok=True)
if __name__ == "__main__":
    df = pd.read_csv(filename, nrows=None).dropna(subset=[content_column])
    parser = AlertDrain(
        hashid=partition_key, savePath=outdir, depth=depth, st=st, regex=regex
    )

    df = pd.concat([df.head(1000)] * 1000)
    s1 = time.time()
    parsed_df = parser.parse(df, col=content_column, drop_duplicate=False)
    s2 = time.time()

    # num_alerts = parsed_df["AlertId"].nunique()
    num_alerts = parsed_df.shape[0]  # parsed_df["AlertId"].nunique()
    time_taken = s2 - s1
    logging.info(f"Parsing {num_alerts} lines of alerts, time taken: {time_taken:.3f}s")

    parsed_df.to_csv(
        output_filename,
        index=False,
        quoting=csv.QUOTE_ALL,
    )
    logging.info("Parsing results are stored in {}".format(output_filename))
