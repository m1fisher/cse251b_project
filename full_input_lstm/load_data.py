import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold

from constants import FUTURE_STEPS

DATA_DIR = "src_data/"
SCALE = 7.0

def load_test_data(data_dir=DATA_DIR):
    test_data = np.load(os.path.join(data_dir, 'test_input.npz'))['data']
    return test_data

def load_train_data(data_dir=DATA_DIR):
    train_data = np.load(os.path.join(data_dir, 'train.npz'))['data']
    train_data = train_data[:, :, :50, :]
    return train_data

def make_dataloaders(scale, data_file, kfold=-1, full_train=False):
    train_data = np.load(data_file)['data']
    N = len(train_data)
    split = []
    if kfold == -1:
        if not full_train:
            # not applying k-fold validation, normally will take the 9:1 split
            val_size = int(0.1 * N)
        else:
            val_size = 0
        train_size = N - val_size
        # TODO: re-enable augmentation
        train_dataset = TrajectoryDatasetTrain(
            train_data[:train_size], scale=SCALE, augment=False
        )
        val_dataset = TrajectoryDatasetValidate(
            train_data[train_size:], scale=SCALE
        )
        split.append([train_dataset, val_dataset])
    else:
        kf = KFold(n_splits=kfold, shuffle=True)
        for train_idx, val_idx in kf.split(train_data):
            train_dataset = TrajectoryDatasetTrain(
                train_data[train_idx], scale=SCALE, augment=False
            )
            val_dataset = TrajectoryDatasetValidate(
                train_data[val_idx], scale=SCALE
            )
            split.append([train_dataset, val_dataset])

    datasets = []
    for train_dataset, val_dataset in split:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=24,
            shuffle=True,
            collate_fn=lambda x: Batch.from_data_list(x),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=24,
            shuffle=False,
            collate_fn=lambda x: Batch.from_data_list(x),
        )
        datasets.append([train_dataloader, val_dataloader])
    return datasets


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = SCALE
        self.augment = augment

    def __len__(self):
        return len(self.data) * 60

    def __getitem__(self, idx):
        scene_idx = idx // 60
        time_idx = min(idx % 60, 60 - FUTURE_STEPS)
        scene = self.data[scene_idx]
        # Getting 50 historical timestamps and future timestamps
        hist = scene[:, time_idx:time_idx+50, :].copy()  # (agents=50, time_seq=50, 6)
        #future = torch.tensor(scene[:, (time_idx+50+FUTURE_STEPS - 1)].copy(), dtype=torch.float32)  # (60, 2)
        future = scene[:, time_idx+50:(time_idx+50+FUTURE_STEPS)].copy()  # (60, 2)

        def wrap(a):
            """Map angle to (-π, π]."""
            return (a + np.pi) % (2 * np.pi) - np.pi

        # TODO: Make sure that data augmentation is correct,
        # consider shifting by some distance.
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
                hist[..., 4] = wrap(hist[..., 4] + theta)
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                hist[..., 4] = wrap(np.pi - hist[..., 4])
                future[:, 0] *= -1

        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, 49, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin
        future[..., :2] = future[..., :2] - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=torch.tensor(future, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

        return data_item

class TrajectoryDatasetValidate(Dataset):
    def __init__(self, data, scale):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = SCALE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        # Getting 50 historical timestamps and 60 future timestamps
        hist = scene[:, :50, :].copy()  # (agents=50, time_seq=50, 6)
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)  # (60, 2)

        def wrap(a):
            """Map angle to (-π, π]."""
            return (a + np.pi) % (2 * np.pi) - np.pi

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
    def __init__(self, data, scale):
        """
        data: Shape (N, 50, 110, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        """
        self.data = data
        self.scale = SCALE

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
