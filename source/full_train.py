### using full training dataset to train the model; intended for final model output

import sys
import yaml
from load_data import DATA_DIR, make_dataloaders, scale
from models import LSTM, LinearForecast
from train import run_training


if __name__ == "__main__":
    model_dir = sys.argv[1]
    with open(f'{model_dir}/model_cfg.yaml') as f:
        cfg = yaml.safe_load(f)
    ## obviously no k-fold validation
    print('Running full training')
    cfg['k_id'] = ''
    t_dataloader, v_dataloader = make_dataloaders(scale, DATA_DIR, full_train=True)[0]  # default is to output one set of loaders
    run_training(cfg, model_dir, t_dataloader, -1)  # -1 signifies not running validation scores
