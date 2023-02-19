import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import csv
import logging
import argparse
import time
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ipack.AIN.model import AINMatcher, MyEarlyStopping
from ipack.AIN.dataloader import process_features, get_dataloader
from ipack.AIN.evaluation import get_eval_df
from ipack.utils import seedEverything, set_logger

# python ticket_event_correlation.py --train --file ../data/train/train_ticket_alert_pairs_parsed.csv --outdir "./ipack_results/ain_results"
# python ticket_event_correlation.py --file ../data/train/train_ticket_alert_pairs_parsed.csv --outdir "./ipack_results/ain_results"

seedEverything()
set_logger("ipack_results/execution.log")

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--outdir", type=str)
params = vars(parser.parse_args())

max_epochs = 1000
patience = 10
embedding_dim = 128

outdir = params["outdir"]
checkpointdir = os.path.join(outdir, "checkpoints")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
os.makedirs(outdir, exist_ok=True)

if __name__ == "__main__":
    sr_feature_used = {
        "sparse": ["TicketService", "TicketProductName", "TicketCategory"],
        "text": ["TicketSummary"],
    }

    incident_feature_used = {
        "sparse": [
            "AlertOwningComponent",
            "AlertOwningService",
            "AlertSeverity",
            "AlertMonitorId",
            "EventId",
        ],
        "text": ["EventTemplate"],
    }

    allfeature_used = {"SR": sr_feature_used, "Incident": incident_feature_used}

    data = pd.read_csv(params["file"], low_memory=False)

    data = pd.concat([data] * 1000).reset_index(drop=True)
    print("# Pairs for correlation: {}".format(data.shape[0]))

    if params["train"]:
        train_data, test_data = data, None
    else:
        train_data, test_data = None, data

    data_values, est, pp = process_features(
        allfeature_used,
        train_data,
        test_data,
        cachedir=checkpointdir,
    )

    sr_feat_names = pp.get_feature_names()["SR"]
    incident_feat_names = pp.get_feature_names()["Incident"]

    data_loader = get_dataloader(
        data_values,
        sr_feat_names,
        incident_feat_names,
        batch_size=1024,
        shuffle=True if params["train"] else False,
        num_workers=1,
    )

    net = AINMatcher(
        vocab_size_dict=est.vocab_size,
        sr_feat_names=sr_feat_names,
        incident_feat_names=incident_feat_names,
        embedding_dim=embedding_dim,
    )

    if params["train"]:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            default_root_dir=checkpointdir,
            callbacks=[
                MyEarlyStopping(monitor="train_loss", mode="min", patience=patience),
                ModelCheckpoint(
                    monitor="train_loss",
                    dirpath=checkpointdir,
                    filename="trained_ain_checkpoint",
                    save_top_k=1,
                    mode="min",
                ),
            ],
        )
        s1 = time.time()
        trainer.fit(model=net, train_dataloaders=data_loader)
        s2 = time.time()
        total_time_train = s2 - s1
        logging.info(f"Training done, total time taken: {total_time_train:.3f}s")
    else:
        model_file = os.path.join(checkpointdir, "trained_ain_checkpoint-v2.ckpt")
        trainer = pl.Trainer(default_root_dir=checkpointdir)
        logging.info("Loading trained AIN model from: {}.".format(model_file))
        model = AINMatcher.load_from_checkpoint(model_file)
        model.eval()

        s1 = time.time()
        predictions_test = trainer.predict(model=model, dataloaders=data_loader)
        s2 = time.time()
        test_time = s2 - s1
        logging.info(f"Testing done, time taken: {test_time:.3f}s")

        test_eval_df = get_eval_df(test_data, predictions_test)
        test_eval_df.to_csv(
            os.path.join(outdir, os.path.basename(params["file"])),
            index=False,
            quoting=csv.QUOTE_ALL,
        )
