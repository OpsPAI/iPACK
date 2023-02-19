import logging
import os
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from ipack.AIN.preprocess import Preprocessor
from ipack.AIN.feature_encoder import FeatureEncoder
from torch.utils.data import DataLoader
from ipack.utils import flatten_dict_values


def get_dataloader(
        df,
        sr_feat_names,
        incident_feat_names,
        batch_size,
        shuffle,
        num_workers=1,
):
    samples = pack_samples(df, sr_feat_names, incident_feat_names)
    loader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


def process_features(feature_used, train_data=None, test_data=None, cachedir='./'):
    os.makedirs(cachedir, exist_ok=True)
    if train_data is not None:
        pp = Preprocessor(feature_used)
        train_df = pp.transform(train_data)
        est = FeatureEncoder(pp.get_feature_names(merge_incident_sr=True))
        train_values = est.fit_transform(train_df)

        with open(os.path.join(cachedir, "preprocessor.pkl"), "wb") as pp_fw, \
                open(os.path.join(cachedir, "feature_encoder.pkl"), "wb") as est_fw:
            pickle.dump(pp, pp_fw)
            pickle.dump(est, est_fw)
            logging.info("Saving feature processor to {}".format(cachedir))
        return train_values, est, pp
    else:
        with open(os.path.join(cachedir, "preprocessor.pkl"), "rb") as pp_fr, \
                open(os.path.join(cachedir, "feature_encoder.pkl"), "rb") as est_fr:
            logging.info("Loading feature processor from {}".format(cachedir))
            pp = pickle.load(pp_fr)
            est = pickle.load(est_fr)
            test_df = pp.transform(test_data)
            test_values = est.transform(test_df)
            return test_values, est, pp


def collate_fn(batch):
    batch_sr_dict = defaultdict(list)
    batch_incident_dict = defaultdict(list)
    batch_label = []

    for (sr_dict, incident_dict, label) in batch:
        for k, v in sr_dict.items():
            batch_sr_dict[k].append(v)
        for k, v in incident_dict.items():
            batch_incident_dict[k].append(v)
        batch_label.append(label)

    batch_sr_dict = {k: torch.Tensor(v) for k, v in batch_sr_dict.items()}
    batch_incident_dict = {k: torch.Tensor(v) for k, v in batch_incident_dict.items()}
    batch_label = torch.FloatTensor(batch_label)
    return batch_sr_dict, batch_incident_dict, batch_label


def pack_samples(df, sr_feat_names, incident_feat_names):
    data = []
    pairs_records = df.to_dict("records")

    for sample_index in tqdm(range(len(pairs_records[0:]))):
        pair = pairs_records[sample_index]
        sr_sample_dict = {feat: pair[feat] for feat in flatten_dict_values(sr_feat_names)}
        incident_sample_dict = {feat: pair[feat] for feat in flatten_dict_values(incident_feat_names)}
        label = pair["Label"]
        sample = (sr_sample_dict, incident_sample_dict, label)
        data.append(sample)
    logging.info("Packing samples as batches done.")

    return data


def get_feature_dict(feat_name_dict, feat_df, id_col="ID"):
    id2feat = {}
    for record in feat_df.to_dict("records"):
        feature = OrderedDict()
        for _, names in feat_name_dict.items():
            feature.update({k: record[k] for k in names})
        id2feat[record[id_col]] = feature
    return id2feat
