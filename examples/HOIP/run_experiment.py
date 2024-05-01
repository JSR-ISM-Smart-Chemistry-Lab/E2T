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
from tqdm import tqdm
from ane.util.crystal import read_cif, get_elem_feats
from ane.util.crystal import n_elem_feats as N_ELEM_FEATS
from ane.materials_property_prediction.gnn import get_gnn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from pymatgen.core.structure import Structure

from enokipy.modeling._episodic import (
    partition_task,
    LightningEpisodicModule,
    LtMetaNetworks,
    TARGET_COL_IDX,
    CustomTaskDataset,
    LabeledRegressionDataset,
)
from enokipy.utils import LOGGER, evaluate_regression_result
from enokipy.utils._visualization import parity_plot, save_loss_series_fig

SEED = 42
np.random.seed(SEED)
SEEDS1 = np.random.randint(0, 1e5, 2)  # seed for train test split
SEEDS2 = np.random.randint(0, 1e5, 2)  # seed for support sampling

MODEL_BASE_PATH = Path(
    "/home/nodak/home/github/EnokiPy/experiments/01_domain_generalization/hoip/episodic/result/"
)

TEST_SIZE = 0.5
FT_SIZES = [0, 10, 20, 30, 40]
MAX_EPOCHS = 300 # 10000
PATIENCE = 100
VAL_RATIO = 0 #0.2
N_MODEL = 10

ELEMENT_POINT = {
    # Cation
    "Ge": 0,
    "Sn": 4,
    "Pb": 8,
    # Anion
    "F": 0,
    "Cl": 1,
    "Br": 2,
    "I": 3,
}

def pyg_collate(data_list):
    return Batch.from_data_list(data_list)

def get_elements_from_cif(cif_path):
    crys = Structure.from_file(cif_path)
    return [elem.name for elem in crys.elements]

def elements_to_category(elements):
    ion_counter = 0
    category = 0
    for elem in elements:
        if elem in ELEMENT_POINT:
            ion_counter += 1
            category += ELEMENT_POINT.get(elem)

    assert ion_counter == 2
    return category


class LtMetaNetworks(LightningEpisodicModule):
    """
    Regression network for episodic training.

    Parameters
    ----------
    encoder : nn.Module
        Encoder module.
    header : nn.Module
        Header module.
    loss : nn.Module, optional
        Loss module, by default nn.MSELoss(reduction="mean").
    **kwargs
        Additional keyword arguments.
    """
    def __init__(
        self,
        gnn_model,
        emb_size,
        normalize_emb,
        header,
        loss=None,
        **kwargs,
    ):
        super(LtMetaNetworks, self).__init__(**kwargs)
        if loss is None:
            loss = nn.MSELoss(reduction="mean")
        self.loss = loss
        self.gnn_model = gnn_model
        self.emb_size = emb_size
        self.normalize_emb = normalize_emb
        self.header = header

        if self.normalize_emb:
            self.layer_norm = nn.LayerNorm(self.emb_size)
            self.featurizer = nn.Sequential(
                get_gnn(N_ELEM_FEATS, n_edge_feats=128, dim_out=self.emb_size, gnn_model=self.gnn_model),
                self.layer_norm,
            )
        else:
            self.featurizer = get_gnn(N_ELEM_FEATS, n_edge_feats=128, dim_out=self.emb_size, gnn_model=self.gnn_model)

        self.save_hyperparameters(logger=False)

    def meta_learn(self, batch):
        self.featurizer.train()
        support, query = batch
        s_embs = self.featurizer(support)
        q_embs = self.featurizer(query)

        # predict
        y_hats = self.header(s_embs, support.y, q_embs)

        return self.loss(y_hats, query.y)

    def predict(self, support, query, scaler=None):
        self.featurizer.eval()
        with torch.no_grad():
            s_embs = self.featurizer(support)
            q_embs = self.featurizer(query)

        y_hat = self.header(s_embs, support.y, q_embs)
        if scaler:
            y_hat = scaler.inverse_transform(y_hat)

        return y_hat


class CIFCustomTaskDataset(CustomTaskDataset):

    def _sample_by_stratified_shuffle_split(self):
        support_nways = self._sample_nways(self.support_nways)
        support_kshots = int(self.support_size / support_nways)
        query_kshots = int(self.query_size / support_nways)

        self.transforms[0].n = support_nways
        self.transforms[0].k = support_kshots + query_kshots

        batch = self[0]
        batch_idx = torch.Tensor(range(batch.y.shape[0]))
        (s_idx, s_label), (q_idx, q_label) = partition_task(batch_idx, torch.tensor(batch.label), support_kshots)
        support = Batch.from_data_list([batch[int(idx)] for idx in s_idx])
        query = Batch.from_data_list([batch[int(idx)] for idx in q_idx])
        return support, query

    def _sample_by_gkfold(self, soft=False):
        # reset filter labels
        self.transforms[0].filter_labels = self.initial_filter_labels

        support_nways = self._sample_nways(self.support_nways)
        query_nways = self._sample_nways(self.query_nways)
        support_kshots = int(self.support_size / support_nways)

        self.transforms[0].n = support_nways
        self.transforms[0].k = support_kshots
        support = self[0]

        if soft is False:
            support_labels = set(support.label)
            query_candidates = list(self.unique_labels - support_labels)
            self.transforms[0].filter_labels = query_candidates

        query_kshots = int(self.query_size / query_nways)
        self.transforms[0].n = query_nways
        self.transforms[0].k = query_kshots
        query = self[0]

        return support, query


class CIFDataset(Dataset):
    def __init__(
            self,
            path,
            id_target_file,
            idx_target,
            n_bond_feats,
            radius=5,
            transform=None,
            pre_transform=None
        ):
        self.path = path
        self.id_target_file = id_target_file
        self.idx_target = idx_target
        self.n_bond_feats = n_bond_feats
        self.radius = radius
        super(CIFDataset, self).__init__(path, transform, pre_transform)
        self.data_list = self.load_dataset()

    def get(self, idx):
        return self.data_list[idx]

    def len(self):
        return len(self.data_list)

    def load_dataset(self):
        elem_feats = get_elem_feats()
        list_cgs = list()
        id_target = np.array(pd.read_excel(self.path + '/' + self.id_target_file))
        id_target = np.hstack([id_target, np.arange(id_target.shape[0]).reshape(-1, 1)])
        targets = id_target[:, self.idx_target]

        gid = 0
        for i in tqdm(range(0, id_target.shape[0])):
            cg = read_cif(elem_feats, self.path, str(id_target[i, 0]), self.n_bond_feats, self.radius, targets[i])
            elements = get_elements_from_cif(self.path + "/" + str(id_target[i, 0]) + ".cif")
            label = elements_to_category(elements)

            if cg is not None:
                cg.gid = gid
                cg.label = label
                list_cgs.append(cg)

                gid += 1

        return list_cgs

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
        return self.trainset, self.testset

    def __len__(self):
        return 1


def finetune_model(
        model,
        train_loader,
        val_loader=None,
        out_dir="./result",
        max_epochs=100, #10000,
        patience=300, #100,
        return_best=True,
    ):
    seed_everything(seed=SEED, workers=True)
    logger = CSVLogger(out_dir)
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
        deterministic=False, #True,
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
        source_data,
        target_data,
        ft_size,
        val_ratio,
        test_size,
        support_size,
        query_size,
        max_epochs,
        patience,
        seed1,
        seed2,
        lr,
        out_dir,
    ):
    # train taskset preparation
    train_idx, test_idx = train_test_split(range(len(target_data)), random_state=seed1, test_size=test_size)
    if ft_size == 0:

        data_test = target_data[test_idx]
        source = Batch.from_data_list(source_data).to(source_model.device)
        test = Batch.from_data_list(data_test).to(source_model.device)
        test_pred = source_model.predict(source, test)
        test_y = test.y

        # evaluation
        result = evaluate_regression_result(test_y, test_pred, test_y, test_pred, out_dir=out_dir)
        result["ft_train_size"] = 0
        result["test_size"] = len(data_test)
        result["seed1"] = seed1
        result["seed2"] = seed2
        result["lr"] = lr

        return result

    ft_train_idx, _ = train_test_split(train_idx, random_state=seed2, train_size=ft_size)
    if val_ratio > 0:
        ft_train_idx, ft_val_idx = train_test_split(ft_train_idx, random_state=seed2, test_size=val_ratio)
        data_ft_val = target_data[ft_val_idx]

    data_ft_train = target_data[ft_train_idx]
    data_test = target_data[test_idx]

    label_train = []
    for data in data_ft_train:
        data.label = 1
        label_train += [1]
        
    for data in source_data:
        data.label = 2
        label_train += [2]

    data_train_concat = data_ft_train + source_data
    
    train_meta_dataset = l2l.data.MetaDataset(
        data_train_concat,
        indices_to_labels={i: label for i , label in enumerate(label_train)}
    )

    # Build Custom Taskdataset
    train_taskset = CIFCustomTaskDataset(
        train_meta_dataset,
        sampling_policy="SoftGKFold",
        support_nways=1,
        support_size=support_size,
        query_nways=1,
        query_size=query_size, #ft_size,
        task_collate=pyg_collate,
    )
    train_epochifier = TrainEpochifier(train_taskset, 10)
    if val_ratio > 0:
        source_train = Batch.from_data_list(data_train_concat)
        val = Batch.from_data_list(data_ft_val)
        val_epochifier = ValEpochifier(source_train, val)
    else:
        val_epochifier = None

    # model preparation
    ft_model = deepcopy(source_model)
    ft_model.lr = lr

    # fine tuning
    ft_model = finetune_model(
        ft_model,
        train_loader=train_epochifier,
        val_loader=val_epochifier,
        out_dir=out_dir,
        max_epochs=max_epochs,
        patience=patience,
        return_best=True if VAL_RATIO > 0 else False,
    )
    
    source = Batch.from_data_list(source_data).to(ft_model.device)
    source_train = Batch.from_data_list(data_train_concat).to(ft_model.device)
    train = Batch.from_data_list(data_ft_train).to(ft_model.device)
    if val_ratio:
        val = Batch.from_data_list(data_ft_val).to(ft_model.device)
        val_y = val.y
        val_pred = ft_model.predict(source_train, val)
        train_val = Batch.from_data_list(data_ft_train + data_ft_val).to(ft_model.device)
        train_val_y = train_val.y
        train_val_pred = ft_model.predict(source_train, train_val)
    test = Batch.from_data_list(data_test).to(ft_model.device)
    
    train_pred = ft_model.predict(source_train, train)
    test_pred = ft_model.predict(source_train, test)
    source_model.to(ft_model.device)
    source_train_pred = source_model.predict(source_train, train)
    source_test_pred = source_model.predict(source_train, test)

    source_y = source.y
    source_pred = ft_model.predict(source_train, source)
    source_source_pred = source_model.predict(source_train, source)
    
    train_y = train.y
    test_y = test.y
    
    # evaluation
    if val_ratio:
        source_train_val_pred = source_model.predict(source_train, train_val)
        source_val_pred = source_model.predict(source_train, val)
        parity_plot(train_y, train_pred, val_y, val_pred, y2_label="val", filename=Path(out_dir) / "train_val.png")
        parity_plot(train_y, source_train_pred, val_y, source_val_pred, y2_label="val", filename=Path(out_dir) / "train_val_source.png")
        parity_plot(train_val_y, source_train_val_pred, test_y, source_test_pred, y2_label="source_model_pred", filename=Path(out_dir) / "parity.png")
        parity_plot(source_y, source_pred, test_y, test_pred, filename=Path(out_dir) / "1_parity_ft.png")
        parity_plot(source_y, source_source_pred, test_y, source_test_pred, filename=Path(out_dir) / "2_parity_source.png")
        result = evaluate_regression_result(train_val_y, train_val_pred, test_y, test_pred, out_dir=out_dir)
        result["ft_val_size"] = len(data_ft_val)
    else:
        parity_plot(train_y, source_train_pred, test_y, source_test_pred, y2_label="source_model_pred", filename=Path(out_dir) / "parity.png")
        parity_plot(source_y, source_pred, test_y, test_pred, filename=Path(out_dir) / "1_parity_ft.png")
        parity_plot(source_y, source_source_pred, test_y, source_test_pred, filename=Path(out_dir) / "2_parity_source.png")
        result = evaluate_regression_result(train_y, train_pred, test_y, test_pred, out_dir=out_dir)
    result["ft_train_size"] = len(data_ft_train) 
    result["test_size"] = len(data_test)
    result["seed1"] = seed1
    result["seed2"] = seed2
    result["lr"] = lr

    return result


if __name__ == "__main__":
    # base_dir = Path(f"./result/patience_{PATIENCE}/")
    base_dir = Path(f"./result/epochs_{MAX_EPOCHS}/")
    summary_file = base_dir / "summary.csv"
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file)
    else:
        df_summary = pd.DataFrame()

    for setting in ("high", "low"):
        source_dataset = CIFDataset(
            path=f"/home/nodak/home/github/ane/ane/materials_property_prediction/datasets/hoip_{setting}/",
            id_target_file="metadata_train.xlsx",
            idx_target=1,
            n_bond_feats=128,
        )
        target_dataset = CIFDataset(
            path=f"/home/nodak/home/github/ane/ane/materials_property_prediction/datasets/hoip_{setting}/",
            id_target_file="metadata_test.xlsx",
            idx_target=1,
            n_bond_feats=128,
        )
        print("================================")
        print(f"Problem setting: {setting}")
        print("================================")
        source_model_paths = list(
            MODEL_BASE_PATH.glob(f"{setting}/LOGO300/version_*/checkpoints/last.ckpt")
        )
        # for model_id, source_model_path in enumerate(source_model_paths):
        for model_id, source_model_path in enumerate(source_model_paths[:N_MODEL]):
            print(source_model_path)
            source_model = LtMetaNetworks.load_from_checkpoint(source_model_path)
            for split_id, (seed1, seed2) in enumerate(itertools.product(SEEDS1, SEEDS2)):
                for ft_size in FT_SIZES:
                    # if len(df_summary) > 0:
                    #     if len(df_summary[
                    #         (df_summary.problem_setting == setting) &
                    #         (df_summary.model_id == model_id) &
                    #         (df_summary.seed1 == seed1) &
                    #         (df_summary.seed2 == seed2) &
                    #         (df_summary.ft_train_val_size == ft_size)
                    #     ]) > 0:
                    #         continue

                    out_dir = base_dir / f"{setting}/model{model_id}/split{split_id}/{ft_size}"
                    if out_dir.exists():
                        continue
                    out_dir.mkdir(exist_ok=True, parents=True)

                    result = finetune_and_eval(
                        source_model=source_model,
                        scaler=None,
                        source_data=source_dataset,
                        target_data=target_dataset,
                        ft_size=ft_size,
                        val_ratio=VAL_RATIO,
                        support_size=10,
                        query_size=10,
                        test_size=TEST_SIZE,
                        max_epochs=MAX_EPOCHS,
                        patience=PATIENCE,
                        seed1=seed1,
                        seed2=seed2,
                        lr=1e-5,
                        out_dir=out_dir,
                    )
                    result["problem_setting"] = setting
                    result["model_id"] = model_id
                    result["ft_train_val_size"] = ft_size
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
