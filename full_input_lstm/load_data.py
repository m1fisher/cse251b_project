import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold

from constants import FUTURE_STEPS, PREV_STEPS, NUM_AGENTS

DATA_DIR = "src_data/"
SCALE = 7.0

def load_test_data(data_dir=DATA_DIR):
    test_data = np.load(os.path.join(data_dir, 'test_input.npz'))['data']
    return test_data

def load_train_data(data_dir=DATA_DIR):
    train_data = np.load(os.path.join(data_dir, 'train.npz'))['data']
    train_data = train_data[:, :, :50, :]
    return train_data

def load_train_data_subset(data_path):
    train_data = np.load(data_path)['data']
    train_data = train_data[:, :, :50, :]
    return train_data

def mask_far_agents(arr, num_to_keep=10):
    # Suppose `arr` is your array of shape (50, 110, 6).
    #   arr[a, t, f]: agent a, timestep t, feature f,
    # and features 0,1 are the (x,y) coordinates.

    # 1) Extract the (x,y) trajectories for all agents:
    positions = arr[..., :2]          # shape (50, 110, 2)

    # 2) Compute, for each agent a, the per‐timestep distance to the ego (a = 0):
    ego_xy = positions[0]             # shape (110, 2)
    diffs   = positions - ego_xy      # broadcasts to (50, 110, 2)
    dists   = np.linalg.norm(diffs, axis=2)  # shape (50, 110)

    # 3) Take the minimum over time for each agent:
    min_dists = dists.min(axis=1)     # shape (50,)

    # 4) Exclude the ego itself from selection by forcing its “min_dist” to ∞:
    min_dists[0] = np.inf

    # 5) Find the 10 non‐ego indices with smallest min‐distance:
    closest_idxs = np.argsort(min_dists)[:num_to_keep]  # array of 10 agent‐indices

    # 6) Build a boolean mask of size 50:
    mask = np.zeros(arr.shape[0], dtype=bool)
    mask[0] = True                  # always keep ego (index 0)
    mask[closest_idxs] = True       # keep the 10 closest non‐ego agents

    # 7) Zero out everything else (in‐place or by creating a copy).
    #arr_masked = arr * mask[:, None, None]

    return arr[mask]

def augment_features(scenes: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Args:
      scenes: (N_agents, T, 6) with [x,y, v_x,v_y, theta, type]
      dt:     timestep duration (e.g. 0.1s)

    Returns:
      (N_agents, T, new_D) array with extra channels:
        [ x, y,
          v_x, v_y,
          theta_sin, theta_cos,
          speed,
          a_x, a_y,
          accel_mag,
          jerk,
          omega (ang. vel),
          type ]
    """
    scenes = mask_far_agents(scenes, num_to_keep=(NUM_AGENTS - 1))
    #return scenes

    N, T, _ = scenes.shape
    pos   = scenes[..., 0:2]     # (N, T, 2)
    vel   = scenes[..., 2:4]     # (N, T, 2)
    theta = scenes[..., 4]       # (N, T)
    typ   = scenes[..., 5:]     # (N, T, 1)

    # 1) speed
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)  # (N, T, 1)

    # 2) acceleration (as before)
    dv    = vel[:, 1:, :] - vel[:, :-1, :]
    accel = np.concatenate([
        np.zeros((N,1,2)),  # zero-pad at t=0
        dv / dt
    ], axis=1)                                            # (N, T, 2)
    accel_mag = np.linalg.norm(accel, axis=-1, keepdims=True)

    # 3) jerk
    da   = accel[:, 1:, :] - accel[:, :-1, :]
    jerk = np.concatenate([
        np.zeros((N,1,2)),
        da / dt
    ], axis=1)                                            # (N, T, 2)
    jerk_mag = np.linalg.norm(jerk, axis=-1, keepdims=True)

    # 4) heading sin/cos
    theta_sin = np.sin(theta)[..., None]  # (N, T, 1)
    theta_cos = np.cos(theta)[..., None]

    # 5) angular velocity ω
    dtheta = theta[:,1:] - theta[:,:-1]
    omega  = np.concatenate([
       np.zeros((N,1)),
       (dtheta / dt)
    ], axis=1)[..., None]                            # (N, T, 1)

    # concatenate everything

    out = np.concatenate([
      pos,           # 2
      vel,           # 2
      theta_sin,     # 1
      theta_cos,     # 1
      speed,         # 1
      accel,         # 2
      accel_mag,     # 1
      jerk,          # 2
      jerk_mag,      # 1
      omega,         # 1
      typ            # 1
    ], axis=-1)

    return out  # shape (N, T, 15)


def make_dataloaders(scale, data_file, kfold=-1, full_train=False):
    train_data = np.load(data_file)['data']
    N = len(train_data)
    split = []
    if kfold == -1:
        if not full_train:
            # not applying k-fold validation, normally will take the 9:1 split
            val_size = int(0.3 * N)
        else:
            val_size = 0
        train_size = N - val_size
        train_dataset = TrajectoryDatasetTrain(
            train_data[:train_size], scale=SCALE, future_steps=FUTURE_STEPS, augment=True
        )
        val_dataset = TrajectoryDatasetValidate(
            train_data[train_size:], scale=SCALE,
        )
        split.append([train_dataset, val_dataset])
    else:
        kf = KFold(n_splits=kfold, shuffle=True)
        for train_idx, val_idx in kf.split(train_data):
            train_dataset = TrajectoryDatasetTrain(
                train_data[train_idx], scale=SCALE, future_steps=FUTURE_STEPS, augment=False
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

def wrap(a):
    """Map angle to (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def augment(scene):
    # Random rotation
    if np.random.rand() < 0.5:
        theta = np.random.uniform(-np.pi, np.pi)
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )
        scene[..., :2] = scene[..., :2] @ R
        scene[..., 2:4] = scene[..., 2:4] @ R
        scene[..., 4] = wrap(scene[..., 4] + theta)

    # Random mirroring (reflection)
    if np.random.rand() < 0.5:
        scene[..., 0] *= -1
        scene[..., 2] *= -1
        scene[..., 4] = wrap(np.pi - scene[..., 4])

    # Random scaling (zoom)
    if np.random.rand() < 0.0:
        scale = np.random.uniform(0.6, 1.4)
        scene[..., :4] *= scale  # scale x, y, vx, vy

#    # Small Gaussian noise to position and velocity
#    if np.random.rand() < 0.5:
#        noise = np.random.normal(0, 0.05, size=scene[..., :4].shape)
#        scene[..., :4] += noise
#
#    # Small random perturbation of heading angle
#    if np.random.rand() < 0.5:
#        scene[..., 4] += np.random.normal(0, 0.05, size=scene[..., 4].shape)
#        scene[..., 4] = wrap(scene[..., 4])
#
#    # Velocity perturbation
#    if np.random.rand() < 0.5:
#        scene[..., 2:4] += np.random.normal(0, 0.1, size=scene[..., 2:4].shape)

    # Random dropout of agents (optional, for multi-agent)
    if np.random.rand() < 0.0:
        agent_mask = np.random.rand(scene.shape[0]) < 0.1  # 20% agents dropped
        scene[agent_mask, ...] = 0.0

    return scene


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale, future_steps, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        print(f"using augment = {augment}")
        self.data = data
        self.scale = scale
        self.augment = augment
        self.future_steps = future_steps
        self.history_steps = PREV_STEPS
        self.timesteps = data.shape[2]  # 110
        self.max_start = self.timesteps - self.history_steps - self.future_steps + 1
        self.total_samples = len(self.data) * self.max_start

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        scene_idx = idx // self.max_start
        time_idx = idx % self.max_start
        scene = self.data[scene_idx]

        # Data augmentation(only for training)
        if self.augment:
            scene = augment(scene)
        scene = augment_features(scene)
        hist = scene[:, time_idx:time_idx+self.history_steps, :].copy()
        future = scene[:, time_idx+self.history_steps:time_idx+self.history_steps+self.future_steps, :].copy()

        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, self.history_steps - 1, :2].copy()  # (2,)
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
        scene = augment_features(scene)
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
        scene = augment_features(scene)
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
