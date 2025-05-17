# cse251b_project
Deep Learning Course Project


Please download the two .npz datasets from Kaggle Page and unzip them into `src_data` directory.
It is git-ignored due to large size.

## Workflow

1. Design a model as desired in `models.py` (or another .py file)
2. Run training via `python3 source/train.py <model_file>.pth`
3. Run test inference via `python3 source/predict.py <model_file>.pth`. Test dataset predictions will save in `submission.csv`.
