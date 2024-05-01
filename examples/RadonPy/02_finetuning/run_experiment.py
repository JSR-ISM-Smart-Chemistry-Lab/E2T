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

from E2T.core import LtMNNs, TableDataSet, CustomTaskDataset
from E2T.utils import evaluate_regression_result, save_loss_series_fig


SUPPORT_SIZE = 20
QUERY_SIZE = 20
TEST_SIZE = 0.5
SEED1 = 42
SEED2 = 4242
LR = 1e-5
FT_SIZES = [0, 20, 50]

SOURCE_MODEL_PATH = "xxx"
SCALER_PATH = "xxx"
DATA_PATH = "../00_sample_data_preparation/PI1070_preprocessed.csv"

TARGET_COL = "xxx"
CLASS_COL = "xxx"

TARGET_CLASS = 8

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
        return self.trainset[:], self.valset[:]

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
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        deterministic=True,
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience),
            checkpoint_callback,
        ],
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
        source_X,
        source_y,
        target_X,
        target_y,
        ft_size,
        support_size,
        query_size,
        test_size,
        seed1,
        seed2,
        lr,
        out_dir,
    ):
    # taskset for finetuning
    target_X_train, target_X_test, target_y_train, target_y_test = \
        train_test_split(target_X, target_y, test_size=test_size, random_state=seed1)

    # without fine-tuning case
    if ft_size == 0:
        X_train = torch.from_numpy(np.array(source_X).astype(np.float32))
        X_test = torch.from_numpy(np.array(target_X_test).astype(np.float32))
        y_test = torch.from_numpy(np.array(target_y_test).astype(np.float32))

        if scaler:
            y_train = torch.tensor(scaler.transform(source_y.reshape(-1, 1)).astype(np.float32))
        else:
            y_train = torch.from_numpy(np.array(source_y).astype(np.float32))

        source_model.to(X_train.device)
        y_test_pred = source_model.predict(X_train, y_train, X_test, scaler=scaler)

        result = evaluate_regression_result(y_test_pred, y_test, out_dir=out_dir)
        result["ft_train_val_size"] = 0
        result["ft_train_size"] = 0
        result["ft_val_size"] = 0
        result["test_size"] = len(target_X_test)
        result["seed1"] = seed1
        result["seed2"] = seed2

        return result

    # fine-tuning case
    X_ft, _, y_ft, _ = \
        train_test_split(target_X_train, target_y_train, train_size=ft_size, random_state=seed2)
    X_ft_train, X_ft_val, y_ft_train, y_ft_val = \
        train_test_split(X_ft, y_ft, test_size=0.2, random_state=seed2)

    X_ft_concat = pd.concat([X_ft_train, source_X], axis=0)
    y_ft_concat = np.concatenate([y_ft_train, source_y.reshape(-1, 1)], axis=0)
    if scaler:
        y_ft_concat = scaler.transform(y_ft_concat)

    ft_train_dataset = TableDataSet(
        X_ft_concat, y_ft_concat, [0] * len(X_ft_train) + [1] * len(source_X),
    )
    ft_val_dataset = TableDataSet(
        X_ft_val, y_ft_val, [0] * len(X_ft_val),
    )

    ft_meta_dataset = l2l.data.MetaDataset(ft_train_dataset)
    ft_taskset = CustomTaskDataset(
        ft_meta_dataset,
        sampling_policy="SoftGKFold",
        support_nways=1,
        support_size=support_size,
        query_nways=1,
        query_size=query_size,
    )

    train_epochifier = TrainEpochifier(ft_taskset, 10)
    val_epochifier = ValEpochifier(ft_train_dataset, ft_val_dataset)

    # model finetuning
    ft_model = deepcopy(source_model)
    ft_model.lr = lr
    ft_model = finetune_model(
        ft_model,
        train_loader=train_epochifier,
        val_loader=val_epochifier,
        out_dir=out_dir,
        return_best=False,
    )

    # evaluation
    X_train = torch.from_numpy(np.array(X_ft_concat).astype(np.float32))
    y_train = torch.from_numpy(np.array(y_ft_concat).astype(np.float32))
    X_test = torch.from_numpy(np.array(target_X_test).astype(np.float32))
    y_test = torch.from_numpy(np.array(target_y_test).astype(np.float32))
    y_test_pred = ft_model.predict(X_train, y_train, X_test, scaler=scaler)

    result = evaluate_regression_result(y_test, y_test_pred, out_dir=out_dir)
    result["ft_train_val_size"] = len(X_ft_concat)
    result["ft_train_size"] = len(X_ft_train)
    result["ft_val_size"] = len(X_ft_val)
    result["support_size"] = support_size
    result["query_size"] = query_size
    result["test_size"] = len(target_X_test)
    result["seed1"] = seed1
    result["seed2"] = seed2
    result["lr"] = lr

    return result


if __name__ == "__main__":

    source_model = LtMNNs.load_from_checkpoint(SOURCE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    data = pd.read_csv(DATA_PATH)

    # harf coding
    source_data = data[data[CLASS_COL]!=TARGET_CLASS]
    target_data = data[data[CLASS_COL]==TARGET_CLASS]
    source_X, source_y = source_data.drop(columns=[TARGET_COL, CLASS_COL]), source_data[TARGET_COL]
    target_X, target_y = target_data.drop(columns=[TARGET_COL, CLASS_COL]), target_data[TARGET_COL]


    results = []
    for ft_size in FT_SIZES:
        out_dir = Path(OUT_DIR) / f"ft_{ft_size}"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = finetune_and_eval(
            source_model=source_model,
            scaler=scaler,
            source_X=source_X,
            source_y=source_y,
            target_X=target_X,
            target_y=target_y,
            ft_size=ft_size,
            support_size=SUPPORT_SIZE,
            query_size=QUERY_SIZE,
            test_size=TEST_SIZE,
            seed1=SEED1,
            seed2=SEED2,
            lr=LR,
            out_dir=out_dir,
        )
        result["source_model_path"] = SOURCE_MODEL_PATH.absolute()

        results.append(result)

    pd.DataFrame(results).to_csv(Path(OUT_DIR) / "results.csv", index=False)
