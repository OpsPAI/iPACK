import os
import random
import logging
import hashlib
import pandas as pd
import numpy as np

# basic random seed
DEFAULT_RANDOM_SEED = 2022


def set_logger(log_file=None):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def hashdf(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[0:8]


def flatten_dict_values(adict):
    res = []  # Result list
    if isinstance(adict, dict):
        for key, val in adict.items():
            res.extend(flatten_dict_values(val))
    elif isinstance(adict, list):
        res = adict
    else:
        raise TypeError("Undefined type for flatten: %s" % type(adict))
    return res


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seedTF(seed=DEFAULT_RANDOM_SEED):
    # tensorflow random seed
    import tensorflow as tf
    tf.random.set_seed(seed)


def seedTorch(seed=DEFAULT_RANDOM_SEED):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + tensorflow + torch
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    print(f"Setting seed as: {seed}")
    seedBasic(seed)
    # seedTF(seed)
    seedTorch(seed)


def get_all_num_nodes(graph_dict):
    count = sum([len(graph.nodes()) for k, graph in graph_dict.items()])
    return count


def get_all_num_edges(graph_dict):
    count = sum([len(graph.edges()) for k, graph in graph_dict.items()])
    return count
