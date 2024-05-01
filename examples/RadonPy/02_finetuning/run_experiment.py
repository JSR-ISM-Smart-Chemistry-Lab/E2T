"""
Experiment file.

Brief instruction

1. Print default config file.
    $ python run_experiment.py fit --print_config > xxx.yaml

2. Update yaml file for your setting

2. Run this file (You can also use inline variables to run.)
    $ python run_experiment.py fit -c xxx.yaml
"""
import itertools
from copy import deepcopy
from pathlib import Path

import learn2learn as l2l
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from enokipy.modeling._episodic import (
    EpisodicDataModuleFromCSV,
    LtMetaNetworks,
    TARGET_COL_IDX,
    CustomTaskDataset,
    LabeledRegressionDataset,
)
from enokipy.utils import LOGGER, evaluate_regression_result
from enokipy.utils._visualization import parity_plot, save_loss_series_fig

SEED = 42
np.random.seed(SEED)
SEEDS1 = np.random.randint(0, 1e5, 3)  # seed for train test split
SEEDS2 = np.random.randint(0, 1e5, 3)  # seed for support sampling

MODEL_BASE_PATH = Path(
    "/home/nodak/home/github/EnokiPy/experiments/01_domain_generalization/radonpy/episodic_temp/result/ecfpct/"
)

TARGET_CLASS_LIST = [
    ("cp", 1),
    ("cp", 2),
    ("cp", 3),
    ("cp", 4),
    ("cp", 5),
    ("cp", 6),
    ("cp", 7),
    ("cp", 8),
    ("cp", 9),
    ("cp", 10),
    ("cp", 11),
    ("cp", 12),
    ("cp", 13),
    ("cp", 14),
    ("cp", 15),
    ("cp", 16),
    ("cp", 18),
    ("cp", 19),
    ("cp", 20),
    ("cp", 21),
    ("ri", 1),
    ("ri", 2),
    ("ri", 3),
    ("ri", 4),
    ("ri", 5),
    ("ri", 6),
    ("ri", 7),
    ("ri", 8),
    ("ri", 9),
    ("ri", 10),
    ("ri", 11),
    ("ri", 12),
    ("ri", 13),
    ("ri", 13),
    ("ri", 14),
    ("ri", 15),
    ("ri", 16),
    ("ri", 18),
    ("ri", 19),
    ("ri", 20),
    ("ri", 21),
]

TEST_SIZE = 0.5
FT_SIZES = [0, 20, 50, 100, 200, 500]


class TrainEpochifier(object):
    def __init__(self, tasks, length):
        self.tasks = tasks
        self.length = length

    def __getitem__(self, *args, **kwargs):
        return self.tasks.sample()

    def __len__(self):
        return self.length


class ValEpochifier(object):
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset

    def __getitem__(self, *args, **kwargs):
        return self.trainset[:], self.testset[:]

    def __len__(self):
        return 1


def finetune_model(
        model,
        train_loader,
        val_loader=None,
        out_dir="./result",
        max_epochs=10000, #10000,
        patience=300,
        return_best=True,
    ):
    seed_everything(seed=SEED, workers=True)
    logger = CSVLogger(out_dir)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        deterministic=True,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
            ),
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
        source_data,
        target_data,
        ft_size,
        support_size,
        query_size,
        test_size,
        seed1,
        seed2,
        lr,
        out_dir,
    ):
    # train taskset preparation
    data_train, data_test = train_test_split(target_data, random_state=seed1, test_size=test_size)

    if ft_size == 0:
        Xs = source_data.iloc[:,:-2]
        ys_scl = scaler.transform(source_data.iloc[:,[TARGET_COL_IDX]])
        X_test = data_test.iloc[:,:-2]
        y_test = data_test.iloc[:,TARGET_COL_IDX]

        Xs = torch.from_numpy(np.array(Xs).astype("float32"))
        ys_scl = torch.from_numpy(np.array(ys_scl).astype("float32"))
        X_test = torch.from_numpy(np.array(X_test).astype("float32"))
        y_test = torch.from_numpy(np.array(y_test).astype("float32"))

        source_model.to(Xs.device)
        y_test_pred = scaler.inverse_transform(source_model.predict(Xs, ys_scl, X_test))

        # evaluation
        result = evaluate_regression_result(y_test, y_test_pred, y_test, y_test_pred, out_dir=out_dir)
        result["ft_train_val_size"] = 0
        result["ft_train_size"] = 0
        result["ft_val_size"] = 0
        result["test_size"] = len(data_test)
        result["seed1"] = seed1
        result["seed2"] = seed2
        result["lr"] = lr

        return result

    data_ft_train_val, _ = train_test_split(data_train, random_state=seed2, train_size=ft_size)
    data_ft_train, data_ft_val = train_test_split(data_ft_train_val, random_state=seed2, test_size=0.2)
    
    data_train_concat = pd.concat([data_ft_train, source_data], axis=0)
    
    ft_train_dataset = LabeledRegressionDataset(
        data_train_concat.iloc[:,:-2],
        scaler.transform(data_train_concat.iloc[:, [TARGET_COL_IDX]]),
        [0] * len(data_ft_train) + [1] * len(source_data),
    )
    ft_val_dataset = LabeledRegressionDataset(
        data_ft_val.iloc[:,:-2],
        scaler.transform(data_ft_val.iloc[:, [TARGET_COL_IDX]]),
        [0] * len(data_ft_val),
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

    # test dataset preparation
    test_dataset = LabeledRegressionDataset(
        data_test.iloc[:,:-2],
        data_test.iloc[:,TARGET_COL_IDX],
        [0] * len(data_test)
    )

    # model preparation
    ft_model = deepcopy(source_model)
    ft_model.lr = lr

    # fine tuning
    ft_model = finetune_model(
        ft_model,
        train_loader=train_epochifier,
        val_loader=val_epochifier,
        out_dir=out_dir,
        return_best=False,
    )

    # data
    Xs, ys_scl = ft_train_dataset.X[:len(data_ft_train)], ft_train_dataset.y[:len(data_ft_train)]
    X_ft_train, y_ft_train = Xs, scaler.inverse_transform(ys_scl)
    X_ft_val, y_ft_val = ft_val_dataset.X, scaler.inverse_transform(ft_val_dataset.y)
    Xq2, yq2 = test_dataset.X, test_dataset.y
    
    print(Xs.shape)
    
    # prediction
    y_ft_train_pred = scaler.inverse_transform(ft_model.predict(Xs, ys_scl, X_ft_train))
    y_ft_val_pred = scaler.inverse_transform(ft_model.predict(Xs, ys_scl, X_ft_val))
    yq2_pred = scaler.inverse_transform(ft_model.predict(Xs, ys_scl, Xq2))
    
    source_model.to(ft_model.device)
    y_ft_train_s_pred = scaler.inverse_transform(source_model.predict(Xs, ys_scl, X_ft_train))
    y_ft_val_s_pred = scaler.inverse_transform(source_model.predict(Xs, ys_scl, X_ft_val))
    yq2_s_pred = scaler.inverse_transform(source_model.predict(Xs, ys_scl, Xq2))

    # evaluation
    parity_plot(y_ft_train, y_ft_train_pred, y_ft_val, y_ft_val_pred, y2_label="val", filename=out_dir / "train_val.png")
    parity_plot(y_ft_train, y_ft_train_s_pred, yq2, yq2_s_pred, y1_label="train", filename=out_dir / "train_test_woft.png")
    parity_plot(y_ft_train, y_ft_train_s_pred, y_ft_val, y_ft_val_s_pred, y2_label="val", filename=out_dir / "train_val_woft.png")
    result = evaluate_regression_result(y_ft_train, y_ft_train_pred, yq2, yq2_pred, out_dir=out_dir)
    result["ft_train_val_size"] = len(data_ft_train_val) 
    result["ft_train_size"] = len(data_ft_train) 
    result["ft_val_size"] = len(data_ft_val)
    result["support_size"] = support_size
    result["query_size"] = query_size
    result["test_size"] = len(data_test)
    result["seed1"] = seed1
    result["seed2"] = seed2
    result["lr"] = lr

    return result


if __name__ == "__main__":
    base_dir = Path("./result")
    summary_file = base_dir / "summary.csv"
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file)
    else:
        df_summary = pd.DataFrame()

    for target, pclass in TARGET_CLASS_LIST:
        target_data = pd.read_csv(
            f"/home/nodak/home/github/radonpy_related/experiments/01_meta_vs_others/data/ecfpct_{target}.csv"
        )
        print("================================")
        print(f"target: {target} / polymer class: {pclass}")
        print("================================")
        source_model_paths = list(
            MODEL_BASE_PATH.glob(f"{target}/wo{pclass}/n2000/*/*/version_0/checkpoints/*.ckpt")
        )
        scaler_paths = list(
            MODEL_BASE_PATH.glob(f"{target}/wo{pclass}/n2000/*/*/version_0/scaler.joblib")
        )
        # for model_id, source_model_path in enumerate(source_model_paths):
        for model_id, source_model_path in enumerate(source_model_paths[:5]):
            source_model = LtMetaNetworks.load_from_checkpoint(source_model_path)
            scaler = joblib.load(scaler_paths[model_id])
            for split_id, (seed1, seed2) in enumerate(itertools.product(SEEDS1, SEEDS2)):
                for ft_size in FT_SIZES:

                    out_dir = base_dir / f"{target}/pclass{pclass}/model{model_id}/split{split_id}/{ft_size}"
                    if out_dir.exists():
                        continue
                    out_dir.mkdir(exist_ok=True, parents=True)

                    result = finetune_and_eval(
                        source_model=source_model,
                        scaler=scaler,
                        source_data=target_data[target_data.polymer_class!=pclass],
                        target_data=target_data[target_data.polymer_class==pclass],
                        ft_size=ft_size,
                        support_size=20, #10,
                        query_size=20, #10,
                        test_size=TEST_SIZE,
                        seed1=seed1,
                        seed2=seed2,
                        lr=1e-5,
                        out_dir=out_dir,
                    )
                    result["target"] = target
                    result["pclass"] = pclass
                    result["model_id"] = model_id
                    result["source_model"] = source_model_path.absolute()

                    with open(out_dir / "result.dat", "w") as f:
                        for k , v in result.items():
                            f.write("%s:\t\t%s\n" % (k, v))

                    if summary_file.exists():
                        df_summary = pd.read_csv(summary_file)
                    else:
                        df_summary = pd.DataFrame()

                    df_summary = pd.concat(
                        [df_summary, pd.DataFrame([result])],
                        axis=0,
                        ignore_index=True,
                    )
                    df_summary.to_csv(summary_file, index=False)
