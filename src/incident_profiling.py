import os
import time
import logging
import pickle
import argparse
import pandas as pd
from ipack.IncidentProfiling import (
    windows2pmi,
    df_sliding_timewindow,
    build_graph,
    prune_graph,
)
from ipack.utils import set_logger


# offline
# python incident_profiling.py --train --file ../data/train/historical_alerts_parsed.csv --outdir "./ipack_results/"

# online
# python incident_profiling.py --file ../data/train/historical_alerts_parsed.csv --outdir "./ipack_results/"


set_logger("ipack_results/execution.log")
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--window_size", default=4, type=int)
parser.add_argument("--step_size", default=1, type=int)
parser.add_argument("--outdir", type=str)
params = vars(parser.parse_args())

if __name__ == "__main__":
    event_data = pd.read_csv(params["file"])

    event_data = pd.concat([event_data] * 1)
    event_data["EventId"] = [i for i in range(event_data["EventId"].shape[0])] 

    # read historical events & conduct sliding windows
    event_window = df_sliding_timewindow(
        event_data,
        time_column="AlertCreateDate",
        event_columns=["EventId"],
        window_size=params["window_size"],
        step_size=params["step_size"],
    )

    num_events = event_data["EventId"].nunique()
    if params["train"]:  # offline mode
        s1 = time.time()
        # compute PMI values
        pair_pmi_dict = windows2pmi(event_window)
        s2 = time.time()
        time_taken = s2 - s1
        logging.info(
            f"Precompute {num_events} lines of events, time taken: {time_taken:.3f}s"
        )

        # stored in a model
        output_file = os.path.join(params["outdir"], "pmi_model.pkl")
        logging.info(
            "Finished. The learned static event relations are stored in {}".format(
                output_file
            )
        )
        with open(output_file, "wb") as fw:
            pickle.dump(pair_pmi_dict, fw)

    else:  # online mode
        with open(os.path.join(params["outdir"], "pmi_model.pkl"), "rb") as fr:
            pair_pmi_dict = pickle.load(fr)
        s1 = time.time()
        graphs = build_graph(pair_pmi_dict, event_window)
        graphs_pruned = {
            k: prune_graph(v, cut_ratio_thre=0.8)[0] for k, v in graphs.items()
        }
        s2 = time.time()
        time_taken = s2 - s1
        logging.info(
            f"Online pruning {num_events} lines of events, time taken: {time_taken:.3f}s"
        )
