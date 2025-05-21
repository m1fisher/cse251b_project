import os
import sys

import numpy as np
import torch
import tqdm
from pathlib import Path
import yaml

from load_data import DATA_DIR, make_dataloaders
from models import LSTM


def get_device():
    # Set device for training speedup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def run_training(cfg):
    model_cfg = cfg['model']
    if model_cfg['name'] == 'lstm':
        model = LSTM()
    else:
        raise ValueError(f"Unknown optimizer {model_cfg['name']}")

    device = get_device()
    model.to(device)
    torch.manual_seed(cfg["training"]['seed'])
    np.random.seed(cfg["training"]['seed'])

    scale = 7.0
    train_dataloader, val_dataloader = make_dataloaders(scale, DATA_DIR)

    patience = cfg["training"]["patience"]
    epochs = cfg["training"]["epochs"]

    opt_cfg = cfg["optimizer"]
    if opt_cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt_cfg["lr"],
                                     betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                                     weight_decay=opt_cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer {opt_cfg['name']}")

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.25
    )  # You can try different schedulers

    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    no_improvement = 0
    for epoch in tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch"):
        # ---- Training ----
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, 60, 2)
            loss = criterion(pred[..., :2], y)  # for models that output all 6-dim, evaluate the loss on only the (x,y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                pred = model(batch)
                y = batch.y.view(batch.num_graphs, 60, 2)
                val_loss += criterion(pred[..., :2], y).item()  # for models that output all 6-dim, evaluate the loss on only the (x,y)

                # show MAE and MSE with unnormalized data
                pred = pred[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                val_mae += torch.nn.L1Loss()(pred, y).item()
                val_mse += torch.nn.MSELoss()(pred, y).item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_mse /= len(val_dataloader)
        scheduler.step()
        # scheduler.step(val_loss)

        tqdm.tqdm.write(
            (f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f}"
             f" | val normalized MSE {val_loss:8.4f}, | val MAE {val_mae:8.4f} | val MSE {val_mse:8.4f}")
        )
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stop!")
                break
    return model


if __name__ == "__main__":
    model_dir = sys.argv[1]
    with open(f'{model_dir}/model_cfg.yaml') as f:
        cfg = yaml.safe_load(f)
    filename = f"{model_dir}.trained_model.pt"
    model = run_training(cfg)
    torch.save(model.state_dict(), filename)

