# cse251b_project
Deep Learning Course Project


Please download the two .npz datasets from Kaggle Page and unzip them into `src_data` directory.
It is git-ignored due to large size.

## Standard Workflow

1. Design a model as desired in `models.py` (or another .py file)
2. Create a `model_cfg.yaml` using the `configs/template_model_cfg.yaml` file, and place this config file inside a new `<model_dir>`
3. Run training via `python3 source/train.py <model_dir>`, all intermediate and final results are stored in this `<model_dir>`
4. (pending update) Run test inference via `python3 source/predict.py <model_file>.pth`. Test dataset predictions will save in `submission.csv`.


## Grid Search

1. Create a `grid_cfg.yaml` using the `configs/template_grid_cfg.yaml` file, and place this config file inside a new `<grid_dir>`
2. Run grid search via `source/grid_search.py <grid_dir>`. It will create all combinations of the params as sub-dir under `<grid_dir>`
3. Collect grid search results inside `source/parse_grid_search.ipynb`
