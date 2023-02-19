import csv
import os
import time
import pickle
import logging
import argparse
import pandas as pd
import subprocess
from tqdm import tqdm
from ipack.utils import set_logger
from ipack.IncidentProfiling import df_sliding_timewindow, build_graph, prune_graph
from ipack.aggregation import assign_cluster

# python online_aggregation.py --file ../data/test/ticket_alert_chunk.csv --outdir "./ipack_results/"

set_logger("ipack_results/execution.log")
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--outdir", type=str)
params = vars(parser.parse_args())
aggregation_outdir = os.path.join(params["outdir"], "aggregation_results")
os.makedirs(aggregation_outdir, exist_ok=True)

if __name__ == "__main__":
    # step1: alert parsing
    logging.info("Parsing: {}".format(params["file"]))
    parsing_return_code = subprocess.call(
        'python alert_parsing.py --file "{}" --content_column "AlertTitle" --outdir "../data/test/"'.format(
            params["file"]
        ),
        shell=True,
    )

    # only reseve the first alert with the same template
    parsed_file = params["file"].replace(".csv", "_parsed.csv")
    chunk_parsed = pd.read_csv(parsed_file)
    chunk_parsed = (
        chunk_parsed.sort_values(by="AlertCreateDate")
        .drop_duplicates(["TicketId", "EventId"], keep="first")
        .reset_index(drop=True)
    )
    chunk_parsed.to_csv(parsed_file, index=False, quoting=csv.QUOTE_ALL)

    # step2: incident profiling
    chunk_windows = df_sliding_timewindow(
        chunk_parsed, "AlertCreateDate", "all", window_size=4, step_size=4
    )
    for df in chunk_windows:
        df["EventId"] = range(df.shape[0])
        
    with open(os.path.join(params["outdir"], "pmi_model.pkl"), "rb") as fr:
        pair_pmi_dict = pickle.load(fr)
    s1 = time.time()
    print("Start pruning")
    graphs = build_graph(pair_pmi_dict, [df["EventId"] for df in chunk_windows])
    graphs_pruned = {
        k: prune_graph(v, cut_ratio_thre=0.8)[0] for k, v in tqdm(graphs.items())
    }
    s2 = time.time()
    print(f"Pruning done, time taken: {s2-s1:.3f}")
    chunk_windows = pd.concat(
        [window.assign(WindowId=widx) for widx, window in enumerate(chunk_windows)]
    )
    chunk_windows.to_csv(parsed_file, index=False, quoting=csv.QUOTE_ALL)

    # step3: ticket-event correlation
    correlation_return_code = subprocess.call(
        'python ticket_event_correlation.py --file "{}" --outdir "./ipack_results/ain_results"'.format(
            parsed_file
        ),
        shell=True,
    )
    chunk_predicted = pd.read_csv(
        os.path.join("./ipack_results/ain_results", os.path.basename(parsed_file))
    )

    # aggregation
    chunk_aggregated = assign_cluster(
        chunk_predicted, graph=graphs_pruned, use_graph=True
    )

    for windowid in chunk_aggregated["WindowId"].unique():
        window = chunk_aggregated[chunk_aggregated["WindowId"] == windowid]
        window.to_csv(
            os.path.join(aggregation_outdir, "cluster-{}.csv".format(windowid)),
            index=False,
            quoting=csv.QUOTE_ALL,
        )
