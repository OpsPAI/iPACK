import math
import logging
import pandas as pd
import numpy as np
import networkx as nx
from kneed import KneeLocator
from itertools import combinations
from collections import Counter
from tqdm import tqdm


def build_graph(pair_pmi_dict, windows, default_pmi=-float("inf"), max_window_len=100):
    G_dict = {}
    for w_idx, window in enumerate(windows):
        if max_window_len and len(window) > max_window_len:
            window = window[:max_window_len]
        pairs = list(combinations(window, 2))

        G = nx.Graph()
        for src, dst in pairs:
            pmi = pair_pmi_dict.get((src, dst), default_pmi)
            G.add_edge(src, dst, pmi=pmi)
        G_dict[w_idx] = G
    return G_dict


def windows2pmi(ordered_windows, mincount=2, max_window_len=500):
    windows = [set(v) for v in ordered_windows]
    logging.info("Processing {} windows..".format(len(windows)))

    logging.info("Counting word..")
    word_count = Counter()
    for window in windows:
        word_count.update(window)
    logging.info("Number of word: {}".format(len(word_count)))

    logging.info("Filtering words in windows:")
    windows = [
        list(filter(lambda x: word_count[x] >= mincount, window)) for window in windows
    ]
    num_window = len(windows)

    logging.info("Generating pairs..")
    pairs = []
    for window in tqdm(windows):
        if len(window) > max_window_len:
            window = window[:max_window_len]
        pairs.append(list(combinations(window, 2)))

    logging.info("Counting pairs..")
    word_pair_count = Counter()
    for pair in tqdm(pairs):
        word_pair_count.update(list(pair))
    logging.info("Number of word pairs: {}".format(len(word_pair_count)))

    logging.info("Computing PMI..")
    pmi_dict = {}
    for key in tqdm(word_pair_count):
        word_i = key[0]
        word_j = key[1]
        pair_count = word_pair_count[key]
        word_freq_i = word_count[word_i]
        word_freq_j = word_count[word_j]
        pmi = math.log(
            (1.0 * pair_count / num_window)
            / (1.0 * word_freq_i * word_freq_j / (num_window * num_window))
        )
        pmi_dict[(word_i, word_j)] = pmi
        pmi_dict[(word_j, word_i)] = pmi
    return pmi_dict


def df_sliding_timewindow(ddf, time_column, event_columns, window_size=4, step_size=1):
    ddf[time_column] = ddf[time_column].map(pd.to_datetime)

    ddf = ddf.sort_values(by=time_column, ascending=True).set_index(time_column)

    start_date, end_date = min(ddf.index), max(ddf.index)

    current_date = start_date
    windows = []

    while current_date < end_date:
        window = []
        window_df = ddf.loc[
            current_date : current_date + pd.Timedelta(hours=window_size)
        ]
        current_date = current_date + pd.Timedelta(hours=step_size)
        if window_df.shape[0] > 1:
            if event_columns != "all":
                for col in event_columns:
                    window.extend(window_df[col].tolist())
                windows.append(window)
            else:
                windows.append(window_df)
    return windows


def search_cut_ratio(pmi_list, use_kneed=True):
    if use_kneed:
        kneedle = KneeLocator(range(len(pmi_list)), pmi_list, S=1.0, curve="concave", direction="increasing")
        return round(kneedle.knee, 3)
    else:
        # simple strategy
        if len(pmi_list) <= 3:
            return 1
        idx2diff = []
        window_len = 1  # len(pmi_list) // 3
        pmi_list = np.array(pmi_list)
        for idx in list(range(window_len, len(pmi_list), window_len)):
            left = pmi_list[0:idx]
            right = pmi_list[idx:]
            diff = np.mean(right) * np.var(right) - np.mean(left) * np.var(left)
            idx2diff.append((idx, diff))
        idx_maxdiff = max(idx2diff, key=lambda x: x[1])[0]
        return idx_maxdiff / len(pmi_list)


def prune_graph(graph_item, cut_ratio_thre=0.1, plot_nodes=[]):
    ratio_dict = {}
    reserved_nodes = []
    for src in graph_item.nodes():
        neighbors = []
        for dst, pmi_dict in graph_item[src].items():
            pmi = pmi_dict["pmi"]
            if pmi >= 0:
                neighbors.append((dst, pmi))
        neighbors = sorted(neighbors, key=lambda x: x[1])
        if neighbors:
            events, pmi_list = list(zip(*neighbors))
            cut_ratio = search_cut_ratio(pmi_list)
            if src in plot_nodes:
                ratio_dict[src] = cut_ratio
            if cut_ratio >= cut_ratio_thre:
                reserved_nodes.append(src)
    graph_item_pruned = graph_item.subgraph(reserved_nodes)
    return graph_item_pruned, ratio_dict
