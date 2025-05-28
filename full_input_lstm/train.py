import os
import sys

import numpy as np
import torch
import tqdm
from pathlib import Path
import yaml

from load_data import DATA_DIR, make_dataloaders, SCALE
from models import LSTM, LinearForecast
from socialnetwork_model import SocialLSTMPredictor
from transformer import CrossAgentTransformerPredictor

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


def run_training(cfg, out_dir, train_dataloader, val_dataloader):
    """
    :param cfg:
    :param out_dir:
    :param train_dataloader:
    :param val_dataloader:  set this to be -1 to run full dataset for training (used for outputing final model)
    :return:
    """
    model_cfg = cfg['model']
    if model_cfg['name'] == 'lstm':
        model = LSTM()
    elif model_cfg['name'] == 'LinearForecast':
        model = LinearForecast()
    elif model_cfg['name'] == 'SN':
        model = SocialLSTMPredictor()
    elif model_cfg['name'] == 'Transformer':
        model = CrossAgentTransformerPredictor(num_features=6)
    else:
        raise ValueError(f"Unknown optimizer {model_cfg['name']}")

    device = get_device()
    model.to(device)
    torch.manual_seed(cfg["training"]['seed'])
    np.random.seed(cfg["training"]['seed'])

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

    ## warmup LR to reduce dead ReLU
    warm_epochs = 3
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=0.75, total_iters=warm_epochs
    )
    main_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, main_sched], milestones=[warm_epochs]
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.7
    # )  # You can try different schedulers

    criterion = torch.nn.MSELoss()

    #scaler = torch.amp.GradScaler()  # allows mixed precision for reduced VRAM usage
    ACC_STEPS = 3  # effective_batch = ACC_STEPS × DataLoader batch; reduced VRAM usage

    fp_write = open(f"{out_dir}/training_epoches.{cfg['k_id']}tsv", 'w')
    fp_write.write("epoch\tlearning_rate\ttrain_loss\tval_loss\tval_mae\tval_mse\n")
    best_val_loss = float("inf")
    no_improvement = 0
    running_loss = 0
    z = 0
    #for epoch in tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch"):
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        print(len(train_dataloader))
        optimizer.zero_grad()
        z = 0
        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc="Batches", unit="batch"), start=0):
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, 50, 6)
            loss = criterion(pred, y) / ACC_STEPS   # scale down
            loss.backward()
            # (optional) gradient clip – scale **before** unscaling
            if (step + 1) % ACC_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * ACC_STEPS                # undo scaling
            z += 1
            #print(train_loss / z)

        train_loss /= len(train_dataloader)

        if isinstance(val_dataloader, int) and val_dataloader == -1:
            ## skipping validation as no validation dataset passed in
            scheduler.step()
            # scheduler.step(val_loss)
            tqdm.tqdm.write(
                f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f}"
            )
            fp_write.write(f"{epoch:03d}\t{optimizer.param_groups[0]['lr']:.6f}\t{train_loss:8.4f}\n")
        else:
            import pdb; pdb.set_trace()

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

            val_loss /= len(val_dataloader)
            val_mae /= len(val_dataloader)
            val_mse /= len(val_dataloader)
            scheduler.step()
            # scheduler.step(val_loss)
            tqdm.tqdm.write(
                (f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f}"
                 f" | val normalized MSE {val_loss:8.4f}, | val MAE {val_mae:8.4f} | val MSE {val_mse:8.4f}")
            )
            fp_write.write(f"{epoch:03d}\t{optimizer.param_groups[0]['lr']:.6f}\t{train_loss:8.4f}\t{val_loss:8.4f}\t{val_mae:8.4f}\t{val_mse:8.4f}\n")
            torch.save(model.state_dict(), f"{out_dir}/current_model.pt")
            if val_loss < best_val_loss - 1e-3:
                best_val_loss = val_loss
                no_improvement = 0
                torch.save(model.state_dict(), f"{out_dir}/best_model.{cfg['k_id']}pt")
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stop!")
                    break

    filename = f"{out_dir}/training_final_model.{cfg['k_id']}pt"
    torch.save(model.state_dict(), filename)
    fp_write.close()


if __name__ == "__main__":
    #torch.cuda.empty_cache()
    model_dir = sys.argv[1]
    with open(f'{model_dir}/model_cfg.yaml') as f:
        cfg = yaml.safe_load(f)
    ## group the kfolds
    if cfg['kfolds'] == -1:
        print('Not running kfolds')
        cfg['k_id'] = ''
        dataf = cfg['dataf']
        t_dataloader, v_dataloader = make_dataloaders(SCALE, dataf)[0]  # default is to output one set of loaders
        run_training(cfg, model_dir, t_dataloader, v_dataloader)
    else:
        dataloaders = make_dataloaders(SCALE, DATA_DIR, kfold=cfg['kfolds'])
        for idx, loaders in enumerate(dataloaders):
            print(f"{idx + 1}-fold started...")
            cfg['k_id'] = f'k{idx + 1}.'
            run_training(cfg, model_dir, loaders[0], loaders[1])
