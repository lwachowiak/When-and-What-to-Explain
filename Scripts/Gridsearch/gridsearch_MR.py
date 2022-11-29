from fastai.callback.wandb import *
import wandb
import datetime
from sklearn.model_selection import ParameterGrid
from TimeSeries_Helpers import *


os.environ["WANDB_SILENT"] = "true"

wandb.login()
my_setup(wandb)


# start time
start_time = datetime.datetime.now()

feature_combinations = [
    {"x-position": True, "y-position": True, "Distance Agent": True,
        "Distance User": True, "AoI": True, "Dilation Left": True, "Dilation Right": True, "AI-position x": True, "AI-position y": True, "User-position x": True, "User-position y": True},
    {"x-position": True, "y-position": True, "Distance Agent": True,
        "Distance User": True, "AoI": True, "Dilation Left": True, "Dilation Right": True, "AI-position x": False, "AI-position y": False, "User-position x": False, "User-position y": False}
]

param_grid = {'n_estimators': [40], 'stride_train': [5, 10, 50, 80], 'stride_eval': [5, 10, 50, 80], 'context_length': [0],
              'oversampling': [True, False], 'undersampling': [True], "features": feature_combinations, "interval_length": [20, 40, 60, 80, 100]}

param_grid = list(ParameterGrid(param_grid))
# remove configs were stride is bigger than the interval length
for grid_config in param_grid:
    if grid_config["stride_train"] > grid_config["interval_length"] or grid_config["stride_eval"] > grid_config["interval_length"]:
        print("Removed config: ", grid_config)
        param_grid.remove(grid_config)

print("\n -----------------------\n Number of interations",
      len(param_grid), "x 5", "\n -----------------------")
for i, grid_config in enumerate(param_grid):
    if True:
        print("Round:", i+1, "of", len(param_grid))
        config_minirocket = AttrDict(
            merged_labels=False,
            threshold=80,
            interval_length=grid_config["interval_length"],
            stride_train=grid_config["stride_train"],
            stride_eval=grid_config["stride_eval"],
            context_length=grid_config["context_length"],
            batch_tfms=[TSStandardize()],     # irrelevant for MiniRocket
            batch_size=64,                    # irrelevant for MiniRocket
            train_ids=[],
            valid_ids=[],
            test_ids=[],
            use_lvl1=True,
            use_lvl2=True,
            verbose=False,
            model="MiniRocket",
            n_estimators=grid_config["n_estimators"],
            oversampling=grid_config["oversampling"],
            undersampling=grid_config["undersampling"],
            features=grid_config["features"]
        )
        cross_validate(val_fold_size=5, config=config_minirocket,
                       group="MR-CV-2510", name=str(grid_config))

# end time
end_time = datetime.datetime.now()
print("\nTime taken:", end_time - start_time)
