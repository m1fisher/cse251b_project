import sys
import yaml
import copy
import itertools
import os
from train import run_training
from load_data import DATA_DIR, make_dataloaders, scale


if __name__ == "__main__":
    grid_search_dir = sys.argv[1]
    with open(f'{grid_search_dir}/grid_cfg.yaml') as f:
        cfg = yaml.safe_load(f)
    sweep_keys = {
        k: v for k, v in cfg["optimizer"].items() if (isinstance(v, list)) and (k != 'name')
    }
    static_key = {
        k: [v] for k, v in cfg["optimizer"].items() if (not isinstance(v, list)) and (k != 'name')
    }
    all_keys = {**sweep_keys, **static_key}  # merge the two
    products = list(itertools.product(*all_keys.values()))
    products = [[float(itr) for itr in params] for params in products]  # convert all param to floats
    dir_names = []
    for params in products:
        dir_str = []
        for key_idx, key in enumerate(all_keys.keys()):
            dir_str.append(f"{key}{params[key_idx]}")
        dir_names.append('_'.join(dir_str))

    for params_idx, params in enumerate(products):
        out_dir = dir_names[params_idx]
        cfg_i = copy.deepcopy(cfg)
        for key_idx, key in enumerate(all_keys.keys()):
            cfg_i['optimizer'][key] = params[key_idx]
        os.makedirs(f"{grid_search_dir}/{out_dir}", exist_ok=True)
        print(f"running ({params_idx + 1} of {len(products)}): {out_dir}")
        if cfg['kfolds'] == -1:
            cfg_i['k_id'] = ''
            t_dataloader, v_dataloader = make_dataloaders(scale, DATA_DIR)[0]  # default is to output one set of loaders
            run_training(cfg_i, f"{grid_search_dir}/{out_dir}", t_dataloader, v_dataloader)
        else:
            dataloaders = make_dataloaders(scale, DATA_DIR, kfold=cfg['kfolds'])
            for idx, loaders in enumerate(dataloaders):
                print(f"{idx + 1}-fold started...")
                cfg_i['k_id'] = f'k{idx + 1}.'
                run_training(cfg_i, f"{grid_search_dir}/{out_dir}", loaders[0], loaders[1])

        with open(f"{grid_search_dir}/{out_dir}/model_cfg.yaml", "w") as file:
            yaml.dump(cfg_i, file, default_flow_style=False)
