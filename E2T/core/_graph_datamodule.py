"""
"""

import learn2learn as l2l
import numpy as np
import pandas as pd
from ane.util.crystal import read_cif, get_elem_feats
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Dataset, Batch
from pymatgen.core.structure import Structure

from ._datamodule import CustomTaskDataset


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

def elements_to_category(elements):
    ion_counter = 0
    category = 0
    for element in elements:
        if element in ELEMENT_POINT:
            ion_counter += 1
            category += ELEMENT_POINT.get(element)

    assert ion_counter == 2
    return category


def pyg_collate(data_list):
    return Batch.from_data_list(data_list)

def get_elements_from_cif(cif_file):
    crys = Structure.from_file(cif_file)
    return [elem.name for elem in crys.elements]


class CIFDataset(Dataset):
    def __init__(
        self,
        path,
        id_target_file,
        idx_target,
        n_bond_feats,
        radius=5,
        transform=None,
        pre_transform=None,
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
        id_target = np.array(pd.read_excel(self.path + "/" + self.id_target_file))
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


class CIFCustomTaskDataset(CustomTaskDataset):

    def _sample_same(self):
        pass

    def _sapmle_by_gkfold(self, soft=False):
        # reset filter labels
        self.transforms[0].filter_labels = self.initial_filter_labels

        support_nways = self._sapmle_nways(self.support_nways)
        query_nways = self._sapmle_nways(self.query_nways)
        support_kshots = int(self.support_size / support_nways)

        self.transform[0].n = support_nways
        self.transfor


class CIFDataModule(LightningDataModule):
    """
    A CIF data module for E2T.
    """
    def __init__(
        self,
        train_path,
        train_id_target_file,
        test_path,
        test_id_target_file,
        idx_target,
        sampling_policy="GKFold",
        support_nways=1,
        support_size=10,
        query_nways=1,
        query_size=10,
        random_seed=42,
        val_ratio=0.2,
        scale_y=False,
        train_epoch_length=1,
        val_epoch_length=1,
    ):
        super().__init__()
        self.train_path = train_path
        self.train_id_target_file = train_id_target_file
        self.test_path = test_path
        self.test_id_target_file = test_id_target_file
        self.idx_target = idx_target
        self.random_seed = random_seed
        self.sampling_policy = sampling_policy
        self.support_nways = support_nways
        self.support_size = support_size
        self.query_nways = query_nways
        self.query_size = query_size
        self.val_ratio = val_ratio
        self.scale_y = scale_y
        self.train_epoch_length = train_epoch_length
        self.val_epoch_length = val_epoch_length

    def setup(self, **kwargs):
        self.train_val_dataset = CIFDataset(
            path=self.train_path,
            id_target_file=self.train_id_target_file,
            idx_target=self.idx_target,
            n_bond_feats=128,
            radius=5,
        )
        self.test_dataset = CIFDataset(
            path=self.test_path,
            id_target_file=self.test_id_target_file,
            idx_target=self.idx_target,
            n_bond_feats=128,
            radius=5,
        )

        label_train = np.array([data.label for data in self.train_val_dataset])
        self.unique_train_labels = np.unique(label_train).size

        # split val data
        if self.val_ratio:
            train_idx, val_idx, label_train, label_val = train_test_split(
                range(len(self.train_val_dataset)),
                label_train,
                test_size=self.val_ratio,
                random_state=self.random_seed,
                stratify=label_train,
            )
            self.train_dataset = self.train_val_dataset[train_idx]
            self.val_dataset = self.train_val_dataset[val_idx]
        else:
            self.train_dataset = self.train_val_dataset

        # scale y
        if self.scale_y:
            self.scaler = StandardScaler()
            y_train = np.array([data.y.item() for data in self.train_dataset]).reshape(-1, 1)
            y_train_scaled = self.scaler.fit_transform(y_train).ravel()
            for data, y_scaled in zip(self.train_dataset, y_train_scaled):
                data.y[0, 0] = y_scaled

            if self.val_ratio:
                y_val = np.array([data.y.item() for data in self.val_dataset]).reshape(-1, 1)
                y_val_scaled = self.scaler.transform(y_val).ravel()

                for data, y_scaled in zip(self.val_dataset, y_val_scaled):
                    data.y[0, 0] = y_scaled

        # Create meta dataset
        self.train_meta_dataset = l2l.data.MetaDataset(
            self.train_dataset,
            indices_to_labels={i: label for i, label in enumerate(label_train)},
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
        return CIFDataModule.epochify(self.train_taskset, self.train_epoch_length)

    def val_dataloader(self):
        if self.val_ratio > 0:
            train = Batch.from_data_list(self.train_dataset)
            val = Batch.from_data_list(self.val_dataset)
            return CIFDataModule.epochify_val(train, val)

        return CIFDataModule.epochify(self.train_taskset, self.train_epoch_length)
