from TimeSeries_Helpers import *

# Feature Importance

feature_combinations = [{
    "x-position": True,
    "y-position": True,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": True,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": True,
    "User-position y": True
}, {
    "x-position": True,
    "y-position": True,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": True,
    "Dilation Left": False,
    "Dilation Right": False,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": True,
    "User-position y": True
}, {
    "x-position": True,
    "y-position": True,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": False,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": True,
    "User-position y": True
}, {
    "x-position": True,
    "y-position": True,
    "Distance Agent": False,
    "Distance User": False,
    "AoI": True,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": True,
    "User-position y": True
}, {
    "x-position": False,
    "y-position": False,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": True,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": True,
    "User-position y": True
},
    {
    "x-position": True,
    "y-position": True,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": True,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": False,
    "User-position y": False
},
    {
    "x-position": True,
    "y-position": True,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": True,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": False,
    "AI-position y": False,
    "User-position x": True,
    "User-position y": True
},
    {
    "x-position": False,
    "y-position": False,
    "Distance Agent": False,
    "Distance User": False,
    "AoI": False,
    "Dilation Left": False,
    "Dilation Right": False,
    "AI-position x": True,
    "AI-position y": True,
    "User-position x": True,
    "User-position y": True
},
    {
    "x-position": True,
    "y-position": True,
    "Distance Agent": True,
    "Distance User": True,
    "AoI": True,
    "Dilation Left": True,
    "Dilation Right": True,
    "AI-position x": False,
    "AI-position y": False,
    "User-position x": False,
    "User-position y": False
}]
run_names = ["all features", "w/o dilation", "w/o AoI", "w/o distances",
             " w/o gaze location", "w/o user position", "w/o agent position", "w/o gaze features", "w/o task features"]
importance_scores = []

for i in range(len(feature_combinations)):
    config_minirocket = AttrDict(
        merged_labels=False,
        threshold=80,
        interval_length=60,
        stride_train=5,
        stride_eval=10,
        context_length=0,
        batch_tfms=[TSStandardize()],     # irrelevant for MiniRocket
        batch_size=64,                    # irrelevant for MiniRocket
        train_ids=list(range(0, 20)),
        valid_ids=[],
        test_ids=list(range(20, 30)),
        use_lvl1=True,
        use_lvl2=True,
        model="MiniRocket",
        n_estimators=40,
        features=feature_combinations[i],
        oversampling=True,
        undersampling=False,
        verbose=False,
    )
    importance_scores.append([])
    for j in range(10):
        models, run_results = train_miniRocket(
            config=config_minirocket, group="Analysis-Feature-Importance", name=run_names[i])
        importance_scores[i].append(run_results["test_vsBaseline_accumulated"])

print("\n\n\nThe feature importantce scores are:\n\n", importance_scores)
