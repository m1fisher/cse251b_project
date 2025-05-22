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

def make_dataloaders(scale, data_dir, kfold=-1, full_train=False):
    train_data = np.load(os.path.join(data_dir, "train.npz"))['data']
    N = len(train_data)
    split = []
    if kfold == -1:
        if not full_train:
            # not applying k-fold validation, normally will take the 9:1 split
            val_size = int(0.1 * N)
        else:
            val_size = 0
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

        # --- Pre-computed constants -------------------------------------------
        self.hist_len = 50  # number of frames in the input window
        self.samples_per_scene = 60  # 110 − hist_len (= 60) (can't +1 because we need the last timestep as y for training)
        self.total_samples = len(data) * self.samples_per_scene  # expand each scene into subscenes

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        ## recompute to get the correct scene ID
        scene_idx = idx // self.samples_per_scene
        offset_t = idx % self.samples_per_scene
        scene = self.data[scene_idx]

        start = offset_t
        end = start + self.hist_len  # exclusive
        hist = scene[:, start:end, :].copy()  # (50 ag, 50 t, 6)
        target = scene[:, end, :].copy()  # (50 ag,     6)

        def wrap(a):
            """Map angle to (-π, π]."""
            return (a + np.pi) % (2 * np.pi) - np.pi

        # Data augmentation(only for training)
        if self.augment:
            # 50 % random planar rotation
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
                target[..., :2] = target[..., :2] @ R
                target[..., 2:4] = target[..., 2:4] @ R
                target[..., 4] = wrap(target[..., 4] + theta)

            # 50 % mirror about the y-axis
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1  # x   → −x
                hist[..., 2] *= -1  # vx  → −vx
                hist[..., 4] = wrap(np.pi - hist[..., 4])
                target[..., 0] *= -1
                target[..., 2] *= -1
                target[..., 4] = wrap(np.pi - target[..., 4])

        # ---------- ego-centric translation & normalisation ---------------
        origin = hist[0, -1, :2].copy()  # ego pos @ t=49
        hist[..., :2] -= origin  # translate positions
        target[..., :2] -= origin
        hist[..., :4] /= self.scale  # scale x,y,vx,vy
        target[..., :4] /= self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),  # (50, 50, 6)
            y=torch.tensor(target, dtype=torch.float32),  # (50, 6)
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
            offset=torch.tensor(offset_t, dtype=torch.long),  # — optional —
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
