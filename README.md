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


## Utils (inside `part3_plots.ipynb`)

1. Given a model, first run `predict.py` (modifying the param in __main__ to run on training data)
to generate the predictions on the training data. Then running `gather_training_loss` in the .ipynb on these predictions will give three
outputs: 1) full label vs. prediction dataframe, 2) a list of (MSE_loss, scene_id), and 3) a dictionary {scene_id: sliced_dataframe}.
Sorting output 2) can help finding scenes where predictions are problematic and device solution.
2. The `plot_scene_pred` function visualizes the label vs. prediction of the ego trajectory, alongside
the trajectory of the surrounding vehicles. This should be very helpful in understanding what exactly went wrong in the 
high loss scenes.

## Models tested

1. Multilayer LSTM: modified directly from benchmark LSTM
   1. Even at 2 layers, since the benchmark model only takes in the ego's information, this model will overfit. At 3
   layers, it achieves about 0.5x the training loss but 1.5x testing loss compared to our current best model (the tuned benchmark with 4096 hidden units).
2. Single-step auto-regressive LSTM: breaks the training sequence into 60 independent steps, the model takes in all agents'
data of the first 50 timestep and predict all agents' data in the next one.
   1. Does not work in the current state
   2. It is quite evident that some form of attention weights need to be given to the other agents, otherwise, the model 
   cannot interpret the relationship between the ego and each of the other agents. I tried running the benchmark with all agents'
   data as input, and the prediction is just horrible. Without an attention-like weight, the model assumes the same relationship
   between the ego and the first agent, the ego with the second agent, etc. Simple solution, such as pre-sorting the
   agents' order based on proximity to the ego, does not work. I think the relationship is just too complex.
3. SocialGAT networked-based encoding: this model first encodes each agent, and then it feeds all agents into a socialGAT network, and lastly it runs through an LSTM
   1. The SocialGAT acts as a form of attention-weight learning between the ego and the other agents
   2. This was suggested by chatgpt and then modified. It kind of works when the model is at the current depth. 
   We are getting very similar training/validation loss compared to the best benchmark model. However, for some reason, the public-test loss is around 11 when submitted to Kaggle
   3. This model uses a lot of VRAM. One optimization was used to break a batch into multiple mini-batches. Modify ACC_STEPS
   in train.py will decrease VRAM usage linearly. This implementation effectively makes batch-size defined in the load_data.py
   more like a mini-batch size. It independently loads ACC_STEPS of mini-batches per batch, while only using VRAM size of the 
   minibatch. The real batch size = mini-batch size * ACC_STEPS
   4. I think this model has potential of working. Some development idea includes better encoding and modifying how the socialGAT
   is handling and associating the different timesteps' data.