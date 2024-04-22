"""
Experiment file.

Brief instruction

1. Print default config file.
    $ python run_experiment.py fit --print_config > xxx.yaml

2. Update yaml file for your setting

2. Run this file (You can also use inline variables to run.)
    $ python run_experiment.py fit -c xxx.yaml
"""
from pathlib import Path
from itertools import product

import learn2learn as l2l
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ane.util.crystal import read_cif, get_elem_feats
from ane.util.crystal import n_elem_feats as N_ELEM_FEATS
from ane.materials_property_prediction.gnn import get_gnn
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from pymatgen.core.structure import Structure

from enokipy.modeling._episodic import (
    CustomTaskDataset,
    LightningEpisodicModule,
    ORIGINAL_LABEL, CUT_LABEL, QCUT_LABEL,
    partition_task,
)
from enokipy.utils import LOGGER, set_directory, evaluate_regression_result
from enokipy.utils._visualization import parity_plot, save_loss_series_fig

EVAL_SUPPORT_SIZES = [-1]

# To separate category
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


class EpisodicCIFDataModule(LightningDataModule):
    """
    A data module for episodic learning from CSV data.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    train_labels : array-like
        Labels used for training.
    sampling_policy : str, optional
        Policy for sampling. Default is "SSS".
    support_nways : int, optional
        Number of ways for the support set. Default is 1.
    support_size : int, optional
        Size of the support set. Default is 10.
    query_nways : int, optional
        Number of ways for the query set. Default is 1.
    query_size : int, optional
        Size of the query set. Default is 10.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    samples_per_label : int, optional
        Number of samples per label. Default is -1.
    label_method : str, optional
        Method to label the data. Default is "source".
    train_bins : int, optional
        Number of bins for training data. Default is 7.
    cut_outlier_bound : tuple, optional
        Bounds for cutting outliers. Default is (0.01, 0.99).
    train_epoch_length : int, optional
        Length of a training epoch. Default is 1.
    val_epoch_length : int, optional
        Length of a validation epoch. Default is 1.
    """
    def __init__(
        self,
        train_path,
        train_id_target_file,
        test_path,
        test_id_target_file,
        idx_target,
        sampling_policy="SSS",
        support_nways=1,
        support_size=10,
        query_nways=1,
        query_size=10,
        random_state=42,
        val_ratio=0.2,
        scale_y=False,
        label_method="source",
        train_bins=7,
        cut_outlier_bound=(0.01, 0.99),
        train_epoch_length=1,
        val_epoch_length=1,
    ):
        super().__init__()
        self.train_path = train_path
        self.train_id_target_file = train_id_target_file
        self.test_path = test_path
        self.test_id_target_file = test_id_target_file
        self.idx_target = idx_target
        self.random_state = random_state
        self.sampling_policy = sampling_policy
        self.support_nways = support_nways
        self.support_size = support_size
        self.query_nways = query_nways
        self.query_size = query_size
        self.val_ratio = val_ratio
        self.scale_y = scale_y
        self.label_method = label_method
        self.train_bins = train_bins
        self.train_epoch_length = train_epoch_length
        self.val_epoch_length = val_epoch_length
        self.cut_outlier_bound = cut_outlier_bound

    def setup(self, **kwargs):
        """Assuming last column of data is target and second last column is label
        """
        # Read data
        self.train_val_dataset = CIFDataset(
            path=self.train_path,
            id_target_file=self.train_id_target_file,
            idx_target=self.idx_target,
            n_bond_feats=128,
            radius=5
        )
        self.test_dataset = CIFDataset(
            path=self.test_path,
            id_target_file=self.test_id_target_file,
            idx_target=self.idx_target,
            n_bond_feats=128,
            radius=5
        )
        
        # label setting
        if self.label_method == ORIGINAL_LABEL:
            label_train: np.array = np.array([data.label for data in self.train_val_dataset]) 
        elif self.label_method == QCUT_LABEL:
            # qcut binning
            y_train = pd.Series([data.y.item() for data in self.train_val_dataset])
            label_train: np.array = pd.qcut(y_train.values.ravel(), self.train_bins, labels=False)
        elif self.label_method == CUT_LABEL:
            y_train = pd.Series([data.y.item() for data in self.train_val_dataset])
            y_train_dummy = pd.Series(y_train)

            # Get bottom/top 1% points
            bottom_outlier_bound = y_train_dummy.quantile(self.cut_outlier_bound[0])
            top_outlier_bound = y_train_dummy.quantile(self.cut_outlier_bound[1])

            # extract outliers data
            bottom_outlier = y_train_dummy[y_train_dummy < bottom_outlier_bound]
            top_outlier = y_train_dummy[y_train_dummy > top_outlier_bound]

            # replace outlier values to bound value
            y_train_dummy.loc[bottom_outlier.index, ] = bottom_outlier_bound
            y_train_dummy.loc[top_outlier.index, ] = top_outlier_bound

            # binning based on treated target value
            label_train: np.array = pd.cut(y_train_dummy.ravel(), self.train_bins, labels=False)
        else:
            raise ValueError("invalid label_method")
        
        for data, label in zip(self.train_val_dataset, label_train):
            data.label = label

        # unique train labels
        self.unique_train_labels = np.unique(label_train).size

        # split val data
        if self.val_ratio:
            train_idx, val_idx, label_train, label_val = \
                train_test_split(range(len(self.train_val_dataset)), label_train, test_size=self.val_ratio, random_state=self.random_state, stratify=label_train)
            self.train_dataset = self.train_val_dataset[train_idx]
            self.val_dataset = self.train_val_dataset[val_idx]
        else:
            self.train_dataset = self.train_val_dataset

        # Scale y
        if self.scale_y:
            self.scaler = StandardScaler()
            y_train = np.array([data.y.item() for data in self.train_dataset]).reshape(-1,1)
            y_train_scl = self.scaler.fit_transform(y_train).ravel()
            for data, y_scaled in zip(self.train_dataset, y_train_scl):
                data.y[0,0] = y_scaled

            if self.val_ratio:
                y_val = np.array([data.y.item() for data in self.val_dataset]).reshape(-1,1)
                y_val_scl = self.scaler.transform(y_val).ravel()

                for data, y_scaled in zip(self.val_dataset, y_val_scl):
                    data.y[0,0] = y_scaled

        # Create meta datset
        self.train_meta_dataset = l2l.data.MetaDataset(
            self.train_dataset,
            indices_to_labels={i: label for i , label in enumerate(label_train)}
        )

        # Build Custom Taskdataset
        self.train_taskset = CIFCustomTaskDataset(
            self.train_meta_dataset,
            sampling_policy=self.sampling_policy,
            support_nways=self.support_nways,
            support_size=self.support_size,
            query_nways=self.query_nways,
            query_size=self.query_size,
            task_collate=pyg_collate,
        )

    @staticmethod
    def epochify(taskset, epoch_length):
        class Epochifier(object):
            def __init__(self, tasks, length):
                self.tasks = tasks
                self.length = length

            def __getitem__(self, *args, **kwargs):
                return self.tasks.sample()

            def __len__(self):
                return self.length

        return Epochifier(taskset, epoch_length)

    @staticmethod
    def epochify_val(trainset, testset):
        class EpochifierVal(object):
            def __init__(self, trainset, testset):
                self.trainset = trainset
                self.testset = testset

            def __getitem__(self, *args, **kwargs):
                return self.trainset, self.testset

            def __len__(self):
                return 1

        return EpochifierVal(trainset, testset)

    def train_dataloader(self):
        return EpisodicCIFDataModule.epochify(self.train_taskset, self.train_epoch_length)

    def val_dataloader(self):
        if self.val_ratio > 0:
            train = Batch.from_data_list(self.train_dataset)
            val = Batch.from_data_list(self.val_dataset)
            return EpisodicCIFDataModule.epochify_val(train, val)

        return EpisodicCIFDataModule.epochify(self.train_taskset, self.train_epoch_length)


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


class CustomLightningCLI(LightningCLI):

    def before_instantiate_classes(self) -> None:
        """ This method is called before instantiation """
        try:
            self.save_config_kwargs["config_filename"] = f"{self.config.fit.trainer.logger.init_args.name}.yaml"
        except:
            LOGGER.info("Output config.yaml")
        else:
            LOGGER.info(f"Output {self.save_config_kwargs['config_filename']}")

    def after_fit(self):
        LOGGER.info("Finish training")
        if self.datamodule.scale_y:
            joblib.dump(self.datamodule.scaler, f"{self.trainer.logger.log_dir}/scaler.joblib")

        out_dir = set_directory(f"{self.trainer.logger.log_dir}/eval", prefix=False)
        self.eval_result(out_dir)

        # for csv logger
        loss_path = f"{self.trainer.logger.log_dir}/metrics.csv"
        if Path(loss_path).exists():
            save_loss_series_fig(loss_path, out_dir, xlabel="Episodes", xfactor=self.datamodule.train_epoch_length)

    def eval_result(self, out_dir):
        all_results = []
        for support_size in EVAL_SUPPORT_SIZES:
            result = self._eval_result(support_size, out_dir=out_dir)
            all_results.append(result)

        # self.datamodule.data_train.to_csv(out_dir / "train_data.csv")
        pd.DataFrame(all_results).to_csv(out_dir / "eval_results.csv", index=False)

    def _eval_result(self, support_size, out_dir):
        """Evaluate model"""
        train = Batch.from_data_list(self.datamodule.train_dataset).to(self.model.device)
        if self.datamodule.val_ratio > 0:
            val = Batch.from_data_list(self.datamodule.val_dataset).to(self.model.device)
        test = Batch.from_data_list(self.datamodule.test_dataset).to(self.model.device)
        train_val = Batch.from_data_list(self.datamodule.train_val_dataset).to(self.model.device)

        if support_size > 0:
            if len(self.datamodule.train_val_dataset) < support_size:
                return dict()
            
            support = DataLoader(self.datamodule.train_val_dataset, batch_size=support_size, shuffle=True)
        else:
            support = Batch.from_data_list(self.datamodule.train_val_dataset).to(self.model.device)

        scaler = self.datamodule.scaler if self.datamodule.scale_y else None
        if scaler:
            support_y = scaler.inverse_transform(support.y)
            train_y = scaler.inverse_transform(train.y)
            test_y = test.y
            train_val_y = scaler.inverse_transform(train_val.y)

            support_pred = self.model.predict(support, support, scaler)
            train_pred = self.model.predict(support, train, scaler)
            test_pred = self.model.predict(support, test, scaler)
            train_val_pred = self.model.predict(support, train_val, scaler)

            if self.datamodule.val_ratio > 0:
                val_y = scaler.inverse_transform(val.y)
                val_pred = self.model.predict(support, val, scaler)
        else:
            support_y = support.y
            train_y = train.y
            test_y = test.y
            train_val_y = train_val.y

            support_pred = self.model.predict(support, support)
            train_pred = self.model.predict(support, train)
            test_pred = self.model.predict(support, test)
            train_val_pred = self.model.predict(support, train_val)

            if self.datamodule.val_ratio > 0:
                val_y = val.y
                val_pred = self.model.predict(support, val)

        out_dir_child = out_dir / f"support_{support_size}"
        parity_plot(support_y, support_pred, train_y, train_pred, y1_label="support", y2_label="train", filename=out_dir_child / "parity_plot_support_train.png")
        parity_plot(support_y, support_pred, train_val_y, train_val_pred, y1_label="support", y2_label="train_val", filename=out_dir_child / "parity_plot_support_train_val.png")
        parity_plot(support_y, support_pred, test_y, test_pred, y1_label="support", y2_label="test", filename=out_dir_child / "parity_plot_support_test.png")
        parity_plot(train_y, train_pred, test_y, test_pred, y1_label="train", y2_label="test", filename=out_dir_child / "parity_plot_train_test.png")
        if self.datamodule.val_ratio > 0:
            parity_plot(support_y, support_pred, val_y, val_pred, y1_label="support", y2_label="val", filename=out_dir_child / "parity_plot_support_val.png")
        result = evaluate_regression_result(train_val_y, train_val_pred, test_y, test_pred, out_dir=out_dir_child)
        result["support_size"] = support_size
        result["rand_id"] = self.datamodule.random_state
        result["train_size"] = len(self.datamodule.train_dataset)
        result["val_size"] = len(self.datamodule.val_dataset) if self.datamodule.val_ratio > 0 else 0
        result["train_val_size"] = result["train_size"] + result["val_size"]
        result["test_size"] = len(self.datamodule.test_dataset)
        result["out_dir"] = out_dir_child

        return result
    

def cli_main():
    cli = CustomLightningCLI(
        model_class=LtMetaNetworks,
        datamodule_class=EpisodicCIFDataModule,
    )

if __name__ == "__main__":
    cli_main()
