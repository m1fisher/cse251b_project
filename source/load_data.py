import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold

DATA_DIR = "src_data/"
scale = 7.0

def load_test_data(data_dir=DATA_DIR):
    test_data  = np.load(os.path.join(DATA_DIR, 'test_input.npz'))['data']
    return test_data

def make_dataloaders(scale, data_dir, kfold=-1):
    train_data = np.load(os.path.join(data_dir, "train.npz"))['data']
    N = len(train_data)
    split = []
    if kfold == -1:
        # not applying k-fold validation, simply taking the 9:1 split
        val_size = int(0.1 * N)
        train_size = N - val_size
        train_dataset = TrajectoryDatasetTrain(
            train_data[:train_size], scale=scale, augment=True
        )
        val_dataset = TrajectoryDatasetTrain(
            train_data[train_size:], scale=scale, augment=False
        )
        split.append([train_dataset, val_dataset])
    else:
        kf = KFold(n_splits=kfold, shuffle=True)
        for train_idx, val_idx in kf.split(train_data):
            train_dataset = TrajectoryDatasetTrain(
                train_data[train_idx], scale=scale, augment=True
            )
            val_dataset = TrajectoryDatasetTrain(
                train_data[val_idx], scale=scale, augment=False
            )
            split.append([train_dataset, val_dataset])

    datasets = []
    for train_dataset, val_dataset in split:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=lambda x: Batch.from_data_list(x),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda x: Batch.from_data_list(x),
        )
        datasets.append([train_dataloader, val_dataloader])
    return datasets


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        # Getting 50 historical timestamps and 60 future timestamps
        hist = scene[:, :50, :].copy()  # (agents=50, time_seq=50, 6)
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)  # (60, 2)

        # Data augmentation(only for training)
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                    dtype=np.float32,
                )
                # Rotate the historical trajectory and future trajectory
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, 49, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin
        future = future - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

        return data_item


class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0):
        """
        data: Shape (N, 50, 110, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        """
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Testing data only contains historical trajectory
        scene = self.data[idx]  # (50, 50, 6)
        hist = scene.copy()

        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item
