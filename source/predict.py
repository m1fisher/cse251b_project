import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import torch

from load_data import TrajectoryDatasetTest, load_test_data, load_train_data
from models import LSTM, LinearForecast
from train import get_device

def load_model(initial_model, model_file, device):
    checkpoint = torch.load(model_file, map_location=device)
    initial_model.load_state_dict(checkpoint)
    initial_model.to(device)
    initial_model.eval()
    return initial_model

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
    train_data = load_train_data()
    device = get_device()
    model_file = sys.argv[1]
    outname = sys.argv[2]
    model = load_model(LSTM(), model_file, device)
    predict(model, device, train_data, outname)
