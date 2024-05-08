"""
"""

from copy import deepcopy
from pathlib import Path

import joblib
import learn2learn as l2l
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch

from E2T.core import LtGraphMNNs, CIFDataset, CIFCustomTaskDataset, pyg_collate
from E2T.utils import evaluate_regression_result, save_loss_series_fig


SUPPORT_SIZE = 20
QUERY_SIZE = 20
TEST_SIZE = 0.5
SEED1 = 42
SEED2 = 4242
LR = 1e-5
FT_SIZES = [0, 10, 20]
MAX_EPOCHS = 300
PATIENCE = None
VAL_RATIO = 0

SOURCE_MODEL_PATH = "../01_domain_generalization/result/example/version_0/checkpoints/{your checkpoint file}"
SCALER_PATH = None
SOURCE_DATA_PATH="{your source data path}"
SOURCE_DATA_FILENAME="metadata_train.xlsx"
TARGET_DATA_PATH="{your target data path}"
TARGET_DATA_FILENAME="metadata_test.xlsx"

OUT_DIR = "./result"


class TrainEpochifier(object):
    def __init__(self, tasks, length):
        self.tasks = tasks
        self.length = length

    def __getitem__(self, *args, **kwargs):
        return self.tasks.sample()

    def __len__(self):
        return self.length


class ValEpochifier(object):
    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset

    def __getitem__(self, *args, **kwargs):
        return self.trainset, self.valset

    def __len__(self):
        return 1


def finetune_model(
        model,
        train_loader,
        val_loader,
        out_dir,
        max_epochs=10000,
        patience=300,
        return_best=False,
        seed=42,
    ):
    seed_everything(seed=seed, workers=True)
    logger = CSVLogger(save_dir=out_dir)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    if val_loader:
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
            ),
            checkpoint_callback,
        ]
    else:
        callbacks = list()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        deterministic=False,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
    save_loss_series_fig(f"{trainer.logger.log_dir}/metrics.csv", out_dir)

    if return_best:
        best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
        return best_model.to(model.device)

    return model


def finetune_and_eval(
        source_model,
        scaler,
        source_dataset,
        target_dataset,
        val_ratio,
        ft_size,
        support_size,
        query_size,
        test_size,
        max_epochs,
        patience,
        seed1,
        seed2,
        lr,
        out_dir,
    ):
    # taskset for finetuning
    train_idx, test_idx = train_test_split(range(len(target_dataset)), random_state=seed1, test_size=test_size)

    if scaler:
        raise("scaler is not supported for now.")

    # without fine-tuning case
    if ft_size == 0:
        data_test = target_dataset[test_idx]

        source = Batch.from_data_list(source_dataset).to(source_model.device)
        test = Batch.from_data_list(data_test).to(source_model.device)
        test_pred = source_model.predict(source, test)
        test_y = test.y

        result = evaluate_regression_result(test_y, test_pred, out_dir=out_dir)
        result["ft_train_size"] = 0
        result["test_size"] = len(data_test)
        result["seed1"] = seed1
        result["seed2"] = seed2
        result["lr"] = lr

        return result

    # fine-tuning case
    ft_train_idx, _ = train_test_split(train_idx, random_state=seed2, train_size=ft_size)
    if val_ratio:
        ft_train_idx, ft_val_idx = train_test_split(ft_train_idx, random_state=seed2, test_size=val_ratio)
        data_ft_val = target_dataset[ft_val_idx]

    data_ft_train = target_dataset[ft_train_idx]
    data_test = target_dataset[test_idx]

    label_train = []
    for data in data_ft_train:
        data.label = 1
        label_train += [1]

    for data in source_dataset:
        data.label = 2
        label_train += [2]

    data_train_concat = data_ft_train + source_dataset

    train_meta_dataset = l2l.data.MetaDataset(
        data_train_concat,
        indices_to_labels={i: label for i, label in enumerate(label_train)}
    )

    # Build Custom Taskdataset
    train_taskset = CIFCustomTaskDataset(
        train_meta_dataset,
        sampling_policy="SoftGKFold",
        support_nways=1,
        support_size=support_size,
        query_nways=1,
        query_size=query_size,
        task_collate=pyg_collate,
    )

    train_epochifier = TrainEpochifier(train_taskset, 10)
    if val_ratio > 0:
        source_train = Batch.from_datalist(data_train_concat)
        val = Batch.from_data_list(data_ft_val)
        val_epochifier = ValEpochifier(source_train, val)
    else:
        val_epochifier = None

    # model finetuning
    ft_model = deepcopy(source_model)
    ft_model.lr = lr
    ft_model = finetune_model(
        ft_model,
        train_loader=train_epochifier,
        val_loader=val_epochifier,
        out_dir=out_dir,
        max_epochs=max_epochs,
        patience=patience,
        return_best=True if val_ratio > 0 else False,
    )

    source_train = Batch.from_data_list(data_train_concat).to(ft_model.device)
    test = Batch.from_data_list(data_test).to(ft_model.device)
    test_pred = ft_model.predict(source_train, test)
    test_y = test.y

    result = evaluate_regression_result(test_y, test_pred, out_dir=out_dir)
    result["ft_train_size"] = len(data_ft_train) 
    result["test_size"] = len(data_test)
    result["seed1"] = seed1
    result["seed2"] = seed2
    result["lr"] = lr

    return result


if __name__ == "__main__":

    source_model = LtGraphMNNs.load_from_checkpoint(SOURCE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH else None

    source_dataset = CIFDataset(
        path=SOURCE_DATA_PATH,
        id_target_file=SOURCE_DATA_FILENAME,
        idx_target=1,
        n_bond_feats=128,

    )
    target_dataset = CIFDataset(
        path=TARGET_DATA_PATH,
        id_target_file=TARGET_DATA_FILENAME,
        idx_target=1,
        n_bond_feats=128,

    )

    results = []
    for ft_size in FT_SIZES:
        out_dir = Path(OUT_DIR) / f"ft_{ft_size}"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = finetune_and_eval(
            source_model=source_model,
            scaler=scaler,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            ft_size=ft_size,
            val_ratio=VAL_RATIO,
            support_size=SUPPORT_SIZE,
            query_size=QUERY_SIZE,
            test_size=TEST_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed1=SEED1,
            seed2=SEED2,
            lr=LR,
            out_dir=out_dir,
        )
        result["source_model_path"] = Path(SOURCE_MODEL_PATH).absolute()

        results.append(result)

    pd.DataFrame(results).to_csv(Path(OUT_DIR) / "results.csv", index=False)
