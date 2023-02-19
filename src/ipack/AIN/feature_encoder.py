import logging

logging.getLogger("requests").setLevel(logging.ERROR)

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from transformers import BertTokenizer, BertModel
from collections import OrderedDict

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logging.info(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def get_feature_dict(feat_name_dict, feat_df, id_values):
    id2feat = {}
    for idx, record in enumerate(feat_df.to_dict("records")):
        feature = OrderedDict()
        for _, names in feat_name_dict.items():
            feature.update({k: record[k] for k in names})
        id2feat[id_values[idx]] = feature
    return id2feat


class FeatureEncoder:
    def __init__(self, feat_cols, use_hash=False):
        self.feat_cols = feat_cols
        self.label_encoder = {}
        self.normalizer = {}
        self.use_hash = use_hash
        self.token2idx = {}

        if "text" in feat_cols and feat_cols["text"]:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.text_model = BertModel.from_pretrained("bert-base-uncased")  # this line will crash running in pycharm
            if torch.cuda.is_available():
                self.text_model.to("cuda")

        self.vocab_size = {}

    def fit_transform(self, train_df):
        ddf = pd.DataFrame()
        for col in self.feat_cols["sparse"]:
            logging.info("Fitting sparse: {}".format(col))
            data = self.transform_sparse(train_df[col])
            ddf[col] = data
            self.vocab_size[col] = data.max() + 2

        for col in self.feat_cols["text"]:
            logging.info("Fitting text: {}".format(col))
            data = self.transform_text(train_df[col])
            ddf[col] = data
        ddf["Label"] = train_df["Label"]

        return ddf

    def transform(self, test_df, ):
        ddf = pd.DataFrame()
        # id_values = test_df[id_col].values
        for col in self.feat_cols["sparse"]:
            logging.info("Fitting sparse: {}".format(col))
            data = self.transform_sparse(test_df[col])
            ddf[col] = data

        for col in self.feat_cols["text"]:
            logging.info("Fitting text: {}".format(col))
            data = self.transform_text(test_df[col])
            ddf[col] = data
        ddf["Label"] = test_df["Label"]

        return ddf

    def transform_text(self, series):
        logging.info("Fetching text embedding for {} unique lines...".format(series.nunique()))

        series_value = series.fillna("").astype(str).tolist()
        text = list(set(series_value))

        if torch.cuda.is_available():
            text_len = len(text)
            batch_size = 10
            output_list = []
            i = 0
            while i < text_len:
                batch = text[i: i + batch_size]
                encoded_input = self.tokenizer(
                    batch, return_tensors="pt", verbose=False, padding=True
                )
                encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}
                output = self.text_model(**encoded_input)["pooler_output"]
                output_list.append(output.detach().cpu().numpy())
                i += batch_size
                if i % 100 == 0:
                    print_gpu_utilization()
                    logging.info("{}/{}".format(i, text_len))
                del encoded_input
                torch.cuda.empty_cache()
            unique_output = np.vstack(output_list).tolist()
            text2embedding = dict(zip(text, unique_output))
            output = np.array(list(map(lambda x: text2embedding[x], series_value))).tolist()
        else:
            output = np.zeros([len(series), 768]).tolist() # for cpu testing only
        logging.info("Fetching {} text embedding done.".format(len(output)))
        return output

    def transform_sparse(self, series):
        series = series.fillna("-1").astype(str)
        feat_name = series.name
        if feat_name not in self.label_encoder:
            oov_value = series.nunique()
            self.label_encoder[feat_name] = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=oov_value
            )
            encoded_series = self.label_encoder[feat_name].fit_transform(
                series.values.reshape(-1, 1)
            )
        else:
            encoded_series = self.label_encoder[feat_name].transform(
                series.values.reshape(-1, 1)
            )
        return encoded_series.reshape(-1).astype(int)
