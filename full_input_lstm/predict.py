import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import torch
import tqdm

from load_data import TrajectoryDatasetTest, load_test_data, load_train_data, load_train_data_subset
from transformer import TwoStageTransformerPredictor, LSTM
from train import get_device


def load_model(initial_model, model_file, device):
    checkpoint = torch.load(model_file, map_location=device)
    initial_model.load_state_dict(checkpoint)
    initial_model.to(device)
    initial_model.eval()
    return initial_model


def AR_predict_test_set(model, device, data, output_name):
    scale = 7.0   # TODO: Make this a saved constant
    dataset = TrajectoryDatasetTest(data, scale=scale)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                             collate_fn=lambda xs: Batch.from_data_list(xs))

    pred_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            batch = batch.to(device)
            B = batch.num_graphs
            hist = batch.x.view(B, 50, 50, NUM_FEATURES).contiguous()  # rolling input
            future_xy_norm = []  # formated output

            for step in range(60):  # predict one step at a time
                batch.x = hist  # assign current history to batch.x
                pred_next = model(batch)  # (B, 50, 6)
                future_xy_norm.append(pred_next[..., 0, :2])
                # Roll the window: drop t-0, append the new prediction
                hist = torch.cat(
                    [hist[:, :, 1:, :],  # t-49 … t-1
                     pred_next[:, :, :1, :]],  # new step at t-0
                    dim=2  # added on the 3rd dimension (t)
                )
            # Reshape by stacking the time steps and the batch
            future_xy_norm = torch.stack(future_xy_norm, dim=1)  # (B, 60, 50, 2)
            scale = batch.scale.view(B, 1, 1, 1)
            origin = batch.origin.view(B, 1, 1, 2)
            future_xy_world = future_xy_norm * scale + origin  # reversing the xy centering at t=50
            pred_list.append(future_xy_world[:, :, 0, :].cpu())  # (B, 60, 2)

    pred_list = torch.cat(pred_list, dim=0).numpy()  # (N,60,2)
    pred_output = pred_list.reshape(-1, 2)  # (N*60, 2)
    output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
    output_df.index.name = 'index'
    output_df.to_csv(output_name, index=True)

def predict(model, device, data, output_name):
    scale = 7.0   # TODO: Make this a saved constant
    dataset = TrajectoryDatasetTest(data, scale=scale)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=lambda xs: Batch.from_data_list(xs))

    pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            # Subset to only x, y predictions for ego agent
            pred_norm = pred_norm[:, 0, :, :2]

            # Reshape the prediction to (N, 60, 2)
            pred = pred_norm * batch.scale.view(-1,1,1) + batch.origin.unsqueeze(1)
            pred_list.append(pred.cpu().numpy())
    pred_list = np.concatenate(pred_list, axis=0)  # (N,60,2)
    pred_output = pred_list.reshape(-1, 2)  # (N*60, 2)
    output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
    output_df.index.name = 'index'
    output_df.to_csv(output_name, index=True)


if __name__ == "__main__":
    test_data = load_test_data()
    # train_data = load_train_data()
    #data = load_train_data_subset('src_data/train.npz')
    data = load_test_data('src_data')
    device = get_device()
    model_file = sys.argv[1]
    outname = sys.argv[2]
    model = load_model(LSTM(), model_file, device)
    model.test_time = True
    predict(model, device, data, outname)
    #AR_predict_test_set(model, device, data, outname)
