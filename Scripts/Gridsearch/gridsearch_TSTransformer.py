from tsai.all import *
import wandb
import datetime
from sklearn.model_selection import ParameterGrid
from TimeSeries_Helpers import *

wandb.login()
my_setup(wandb)


# start time
start_time = datetime.datetime.now()

feature_combinations = [
    # {"x-position": True, "y-position": True, "Distance Agent": True,
    #     "Distance User": True, "AoI": True, "Dilation Left": True, "Dilation Right": True,  "AI-position x": True, "AI-position y": True, "User-position x": True, "User-position y": True},
    # {"x-position": True, "y-position": True, "Distance Agent": True,
    #    "Distance User": True, "AoI": True, "Dilation Left": True, "Dilation Right": True,  "AI-position x": False, "AI-position y": False, "User-position x": False, "User-position y": False},
    {"x-position": True, "y-position": True, "Distance Agent": True,
     "Distance User": True, "AoI": False, "Dilation Left": False, "Dilation Right": False,  "AI-position x": True, "AI-position y": True, "User-position x": True, "User-position y": True},
    {"x-position": True, "y-position": True, "Distance Agent": True,
     "Distance User": True, "AoI": False, "Dilation Left": False, "Dilation Right": False,  "AI-position x": False, "AI-position y": False, "User-position x": False, "User-position y": False}
]

param_grid = {'n_epoch': [30],
              'dropout_TST': [0, 0.8],
              'fc_dropout_TST': [0.2, 0.3],
              'stride_train': [5, 10, 50, 80],
              'stride_eval': [5, 10, 50, 80],
              'lr': [2e-4],
              'features': feature_combinations,
              'focal_loss': [False, True],
              "interval_length": [20, 40, 60, 80, 100],
              "context_length": [0],
              'oversampling': [True, False],
              "batch-size": [64, 128],
              "batch_tfms": [None]}

param_grid = list(ParameterGrid(param_grid))

# remove configs were stride is bigger than the interval length
for grid_config in param_grid:
    if grid_config["stride_train"] > grid_config["interval_length"] or grid_config["stride_eval"] > grid_config["interval_length"]:
        param_grid.remove(grid_config)

print("\n -----------------------\n Number of interations",
      len(param_grid), "x 5", "\n -----------------------")


for i, grid_config in enumerate(param_grid):
    if True:
        print("Round:", i+1, "of", len(param_grid))
        print(grid_config)
        config = AttrDict(
            merged_labels=False,
            threshold=80,
            interval_length=grid_config["interval_length"],
            stride_train=grid_config["stride_train"],
            stride_eval=grid_config["stride_eval"],
            context_length=grid_config['context_length'],
            train_ids=[],
            valid_ids=[],
            test_ids=[],
            use_lvl1=True,
            use_lvl2=True,
            model="TST",
            lr=grid_config["lr"],
            n_epoch=grid_config["n_epoch"],
            dropout_TST=grid_config["dropout_TST"],
            fc_dropout_TST=grid_config["fc_dropout_TST"],
            batch_tfms=grid_config["batch_tfms"],
            batch_size=grid_config["batch-size"],
            focal_loss=grid_config["focal_loss"],
            features=grid_config["features"],
            oversampling=grid_config["oversampling"],
            undersampling=False,
            verbose=False,
        )

        cross_validate(val_fold_size=5, config=config,
                       group=config.model+"-CV-1031", name=str(grid_config))


end_time = datetime.datetime.now()
print("Time taken:", end_time - start_time)
