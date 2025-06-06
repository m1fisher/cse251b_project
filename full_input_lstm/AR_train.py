import os
import sys

import numpy as np
import torch
import tqdm
from pathlib import Path
import yaml

from constants import FUTURE_STEPS, NUM_AGENTS, NUM_FEATURES
from load_data import DATA_DIR, make_dataloaders, SCALE
from socialnetwork_model import SocialLSTMPredictor
from transformer import TwoStageTransformerPredictor, LSTM
from ssm import KFLSTMForecast, SSM_LSTM_MultiForecast
from TGR import TFMFModel

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
        #model = GRU()
    elif model_cfg['name'] == 'LinearForecast':
        model = LinearForecast()
    elif model_cfg['name'] == 'SN':
        model = SocialLSTMPredictor()
    elif model_cfg['name'] == 'Transformer':
        #model = CrossAgentTransformerPredictor(num_features=6)
        #model = AutoRegressiveMLP(num_features=6)
        #model = AgentOnlyTransformerPredictor(num_features=6)
        model = TwoStageTransformerPredictor(num_features=NUM_FEATURES, future_steps=FUTURE_STEPS)
        model = GNNLSTMPredictor()
    elif model_cfg['name'] == 'TGR':
        model = TFMFModel()
    elif model_cfg['name'] == 'SSM':
        model = SSM_LSTM_MultiForecast()
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
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=opt_cfg["lr"],
                                     betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                                     weight_decay=opt_cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer {opt_cfg['name']}")

    base_lr = opt_cfg["lr"]
    from torch.optim.lr_scheduler import OneCycleLR
#    scheduler = OneCycleLR(
#        optimizer,
#        max_lr=base_lr * 10,   # up-and-down mountain: peak around 10× base
#        total_steps=epochs * len(train_dataloader),
#        pct_start=0.3,          # fraction of cycle spent increasing
#        anneal_strategy='cos',  # cosine anneal down
#    )
    ## warmup LR to reduce dead ReLU
    warm_epochs = 5
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_epochs
    )
    main_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.7
    )
#    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
#        optimizer,
#        T_max=epochs,    # number of epochs to anneal over
#        eta_min=1e-6         # minimum learning rate
#    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, main_sched], milestones=[warm_epochs]
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.7
    # )  # You can try different schedulers

    criterion = torch.nn.MSELoss()
    validation_criterion = torch.nn.MSELoss()
    #scaler = torch.amp.GradScaler()  # allows mixed precision for reduced VRAM usage
    ACC_STEPS = 3  # effective_batch = ACC_STEPS × DataLoader batch; reduced VRAM usage

    fp_write = open(f"{out_dir}/training_epoches.{cfg['k_id']}tsv", 'w')
    fp_write.write("epoch\tlearning_rate\ttrain_loss\tval_loss\tval_mae\tval_mse\n")
    best_val_loss = float("inf")
    no_improvement = 0
    running_loss = 0
    z = 0
    #for epoch in tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch"):
    prev_train_loss = float("inf")
    train_loss_improvement_threshold = 0.0003
    train_loss_patience = 3
    train_loss_no_improvement = 0
    lr_scaling_factor = 0.5
    lr_threshold_scaling_factor = 0.75
    min_lr = 1e-6
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        z = 0
        train_loss = 0

        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc="Batches", unit="batch"), start=0):
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, NUM_AGENTS, FUTURE_STEPS, NUM_FEATURES)
            # TODO: try all loss variations
            #pred[..., :2] = pred[..., :2] * batch.scale.view(batch.num_graphs, 1, 1, 1) + batch.origin.view(batch.num_graphs, 1, 1, 2)
            #y[..., :2] = y[..., :2] * batch.scale.view(batch.num_graphs, 1, 1, 1) + batch.origin.view(batch.num_graphs, 1, 1, 2)

            # dynamically assign 1/2 loss from ego, 1/2 from other agents


            loss_ego = criterion(pred[:, 0, :, :2], y[:, 0, :, :2])
            loss = loss_ego
#            loss_others = torch.sqrt(criterion(pred[:, 1:, :, :], y[:, 1:, :, :]))
#            loss = loss_ego  #+ 0.01 * loss_others
#            L0 = loss_ego.detach()
#            L1 = loss_others.detach()
#            eps = 1e-8
#            w0 =   L1 / (L0 + L1 + eps)   # if others-loss ≫ agent0-loss, w0 → 1
#            w1 =   L0 / (L0 + L1 + eps)   # if others-loss ≫ agent0-loss, w1 → 0
#            #alpha = 0.9
#            loss = w0*loss_ego +w1*loss_others

            loss /= ACC_STEPS   # scale down

            # 2) smoothness loss (only if future_steps ≥ 3)
            lambda_ = 5e-1
            if pred.size(2) >= 3:
                acc = pred[..., 2:, :] - 2*pred[..., 1:-1, :] + pred[..., :-2, :]
                L_smooth = acc.pow(2).mean()
            else:
                L_smooth = 0.0

            loss += lambda_ * L_smooth
            loss.backward()
            # (optional) gradient clip – scale **before** unscaling
            if (step + 1) % ACC_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * ACC_STEPS                # undo scaling
            z += 1
            if z > 5000:
                break
            #print(train_loss / z)
        train_loss = train_loss / z
        if prev_train_loss - train_loss <= train_loss_improvement_threshold:
          train_loss_no_improvement += 1
          print(f'no improvement: {train_loss_no_improvement}; {prev_train_loss} -- {train_loss}')
        else:
            print(f'improved: {prev_train_loss} -> {train_loss}')
            train_loss_no_improvement = 0
        prev_train_loss = train_loss
        if False and train_loss_no_improvement >= train_loss_patience:
            print('lr updated')
            train_loss_no_improvement = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * lr_scaling_factor, min_lr)
                train_loss_improvement_threshold *= lr_threshold_scaling_factor

        if isinstance(val_dataloader, int) and val_dataloader == -1:
            ## skipping validation as no validation dataset passed in
            scheduler.step()
            tqdm.tqdm.write(
                f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f}"
            )
            fp_write.write(f"{epoch:03d}\t{optimizer.param_groups[0]['lr']:.6f}\t{train_loss:8.4f}\n")
        else:
            # ---- Validation ----
            model.eval()
            val_loss = 0
            val_mae = 0
            val_mse = 0
            with torch.no_grad():
                i = 0
                for batch in tqdm.tqdm(val_dataloader):
                    i += 1
                    if i > 1000:
                        break
                    batch = batch.to(device)
                    x = batch.x.view(batch.num_graphs, NUM_AGENTS, 50, NUM_FEATURES)
                    if FUTURE_STEPS == 60:
                        pred = model(x)
                    else:
                        # use autoregression
                        # TODO: set up variable future_step autoregression?
                        # preds = []
                        # for i in range(60):
                        #     pred = model(x)
                        #     preds.append(pred[:, :, 0, :])
                        #     x = torch.cat([x, pred], dim=2)[:, :, 1:]
                        # pred = torch.stack(preds, dim=2)
                        total_future = 60  # or whatever your total prediction length is
                        pred_steps = FUTURE_STEPS  # number of steps your model predicts per call
                        preds = []
                        n_iters = total_future // pred_steps
                        remainder = total_future % pred_steps

                        for _ in range(n_iters):
                            pred = model(x)  # (B, A, pred_steps, F)
                            preds.append(pred)
                            # Concatenate predictions to input, remove oldest steps to keep input length fixed
                            x = torch.cat([x, pred], dim=2)[:, :, pred_steps:]

                        if remainder > 0:
                            pred = model(x)  # (B, A, pred_steps, F)
                            preds.append(pred[:, :, :remainder, :])  # Only take the needed steps

                        # Concatenate all predicted steps along the time axis
                        pred = torch.cat(preds, dim=2)  # (B, A, total_future, F)
                    y = batch.y.view(batch.num_graphs, 60, 2)

                    val_loss += validation_criterion(pred[:, 0, :, :2], y).item()  # for models that output all 6-dim, evaluate the loss on only the (x,y)

                    # show MAE and MSE with unnormalized data
                    pred = pred[:, 0, :, :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                    y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                    val_mae += torch.nn.L1Loss()(pred, y).item()
                    val_mse += torch.nn.MSELoss()(pred, y).item()

            val_loss /= i
            val_mae /= i
            val_mse /= i
            scheduler.step()
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
