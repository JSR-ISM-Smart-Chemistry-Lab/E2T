"""
Datamodule for E2T.
Tabular data is expected to be in a csv file.
"""

import random

import learn2learn as l2l
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Sampling policy
GKFOLD = "GKFOLD"
SOFT_GKFOLD = "SOFT_GKFOLD"
SAME = "SAME"


class TableDataModule(LightningDataModule):
    """
    A data module for E2T.
    """
    def __init__(
        self,
        csv_path,
        train_classes,
        sampling_policy="SSS",
        support_nways=1,
        support_size=10,
        query_nways=1,
        query_size=10,
        random_seed=42,
        val_ratio=0.2,
        scale_y=False,
        samples_per_class=1,
        target_col_idx=-2,
        class_col_idx=-1,
        train_epoch_length=1,
        val_epoch_length=1,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.train_classes = train_classes
        self.sampling_policy = sampling_policy
        self.support_nways = support_nways
        self.support_size = support_size
        self.query_nways = query_nways
        self.query_size = query_size
        self.random_seed = random_seed
        self.val_ratio = val_ratio
        self.scale_y = scale_y
        self.samples_per_class = samples_per_class
        self.target_col_idx = target_col_idx
        self.class_col_idx = class_col_idx
        self.train_epoch_length = train_epoch_length
        self.val_epoch_length = val_epoch_length

    def setup(self, **kwargs):
        """
        Setup the data module.
        """
        origin_data = pd.read_csv(self.csv_path)

        # Sampling training data
        self.class_col = origin_data.columns[self.class_col_idx]
        self.target_col = origin_data.columns[self.target_col_idx]
        self.train_data = origin_data[origin_data[self.class_col].isin(self.train_classes)]
        self.test_data = origin_data[~origin_data[self.class_col].isin(self.train_classes)]

        if self.samples_per_class == -1:
            self.train_val_data_sampled = self.train_data
        else:
            try:
                self.train_val_data_sampled = self.train_data.groupby(self.class_col).sample(self.samples_per_class, random_state=self.random_seed)
            except ValueError:
                n_samples = self.samples_per_class * self.train_data[self.class_col].nunique()
                self.train_val_data_sampled = self.train_data.sample(n_samples, random_state=self.random_seed)

        # unique train labels
        self.unique_train_labels = self.train_val_data_sampled[self.class_col].unique()

        # Split validation data
        if self.val_ratio > 0:
            self.train_data_sampled, self.val_data_sampled = train_test_split(
                self.train_val_data_sampled, test_size=self.val_ratio, random_state=self.random_seed
            )
        else:
            self.train_data_sampled = self.train_val_data_sampled
            self.val_data_sampled = None

        # Split columns for train
        X_train, y_train, class_train = (
            self.train_data_sampled.drop(columns=[self.class_col, self.target_col]),
            self.train_data_sampled[self.target_col],
            self.train_data_sampled[self.class_col],
        )

        # scale y
        if self.scale_y:
            self.scaler = StandardScaler()
            y_train_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1))
            self.train_dataset = TableDataset(X_train, y_train_scaled, class_train)
        else:
            self.train_dataset = TableDataset(X_train, y_train, class_train)

        # Split columns for validation
        if self.val_data_sampled is not None:
            X_val, y_val, class_val = (
                self.val_data_sampled.drop(columns=[self.class_col, self.target_col]),
                self.val_data_sampled[self.target_col],
                self.val_data_sampled[self.class_col],
            )
            if self.scale_y:
                y_val_scaled = self.scaler.transform(y_val.values.reshape(-1, 1))
                self.val_dataset = TableDataset(X_val, y_val_scaled, class_val)
            else:
                self.val_dataset = TableDataset(X_val, y_val, class_val)

        self.train_meta_dataset = l2l.data.MetaDataset(self.train_dataset)

        self.train_taskset = CustomTaskDataset(
            self.train_meta_dataset,
            sampling_policy=self.sampling_policy,
            support_nways=self.support_nways,
            support_size=self.support_size,
            query_nways=self.query_nways,
            query_size=self.query_size,
            filter_classes=None, # already filtered in setup
        )

    @staticmethod
    def epochify(taskset, epoch_length):
        """
        Epochify the taskset.
        """
        class Epochifier():
            def __init__(self, tasks, epoch_length):
                self.tasks = tasks
                self.epoch_length = epoch_length

            def __len__(self):
                return self.epoch_length

            def __getitem__(self, *args, **kwargs):
                return self.tasks.sample()

        return Epochifier(taskset, epoch_length)

    @staticmethod
    def epochify_val(trainset, taskset):
        """
        Epochify the taskset for validation.
        """
        class EpochifierVal():
            def __init__(self, trainset, taskset):
                self.trainset = trainset
                self.taskset = taskset

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.trainset[:], self.taskset[:]

        return EpochifierVal(trainset, taskset)

    def train_dataloader(self):
        return TableDataModule.epochify(self.train_taskset, self.train_epoch_length)

    def val_dataloader(self):
        if self.val_ratio > 0:
            return TableDataModule.epochify_val(self.train_dataset, self.val_dataset)
        else:
            return TableDataModule.epochify(self.train_taskset, self.train_epoch_length)


class TableDataset(Dataset):
    """
    A dataset for regression tasks with class.
    """
    def __init__(self, X, y, class_labels):
        self.X = torch.from_numpy(np.array(X)).float()
        self.y = torch.from_numpy(np.array(y)).float()
        self.class_labels = torch.from_numpy(np.array(class_labels)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # This order is important for l2l.data.MetaDataset
        return self.X[idx], self.class_labels[idx], self.y[idx]


class CustomTaskDataset(l2l.data.TaskDataset):
    """
    Custom task dataset for sampling tasks with different policies.
    """
    def __init__(
        self,
        dataset,
        sampling_policy="SSS",
        support_nways=1,
        support_size=10,
        query_nways=1,
        query_size=10,
        filter_classes=None,
    ):
        self.transforms = [
            l2l.data.transforms.FusedNWaysKShots(
                dataset,
                n=1,   # dummy value, updated in sample()
                k=1,   # dummy value, updated in sample()
                replacement=True,
                filter_labels=filter_classes,
            ),
            l2l.data.transforms.LoadData(dataset),
        ]
        super(CustomTaskDataset, self).__init__(
            dataset,
            task_transforms=self.transforms,
        )

        # argument mapping
        if sampling_policy.lower() in ["groupkfold", "gkfold"]:
            sampling_policy = GKFOLD
        elif sampling_policy.lower() in ["softgroupkfold", "softgkfold"]:
            sampling_policy = SOFT_GKFOLD
        elif sampling_policy.lower() == "same":
            sampling_policy = SAME
        else:
            raise ValueError(f"Sampling policy {sampling_policy} is not supported.")

        self.sampling_policy = sampling_policy
        self.support_nways = support_nways
        self.support_size = support_size
        self.query_nways = query_nways
        self.query_size = query_size

        self.label_set = set(dataset.labels)
        self.initial_filter_labels = filter_classes if filter_classes else list(self.label_set)

    def sample(self):
        """
        Sample a task.
        """
        if self.sampling_policy == GKFOLD:
            return self._sample_by_gkfold(soft=False)
        elif self.sampling_policy == SOFT_GKFOLD:
            return self._sample_by_gkfold(soft=True)
        elif self.sampling_policy == SAME:
            return self._sample_same()

    @staticmethod
    def _sample_nways(nways):
        if isinstance(nways, (tuple, list)):
            assert len(nways) == 2
            sampled_nways = random.randint(nways[0], nways[1])
        elif isinstance(nways, int):
            sampled_nways = nways
        else:
            raise ValueError(f"nways should be int or tuple or list, got {type(nways)}")
        return sampled_nways

    def _sample_by_gkfold(self, soft=False):
        """
        Sample tasks by group k-fold.

        Parameters
        ----------
        soft : bool
            If True, sample query classes that are not in the support set.
        """
        # reset filter labels
        self.transforms[0].filter_labels = self.initial_filter_labels

        support_nways = self._sample_nways(self.support_nways)
        query_nways = self._sample_nways(self.query_nways)
        support_kshots = int(self.support_size / support_nways)

        self.transforms[0].n = support_nways
        self.transforms[0].k = support_kshots
        support = self[0]

        if soft is False:
            support_labels = set(support[1].tolist())
            query_candidates = list(self.unique_labels - support_labels)
            self.transforms[0].filter_labels = query_candidates

        query_kshots = int(self.query_size / query_nways)
        self.transforms[0].n = query_nways
        self.transforms[0].k = query_kshots
        query = self[0]

        return support, query

    def _sample_same(self):
        """
        Sample the same task for support and query.
        """
        # reset filter labels
        self.transforms[0].filter_labels = self.initial_filter_labels

        support_nways = self._sample_nways(self.support_nways)
        support_kshots = int(self.support_size / support_nways)
        self.transforms[0].n = support_nways
        self.transforms[0].k = support_kshots
        support = self[0]
        return support, support
