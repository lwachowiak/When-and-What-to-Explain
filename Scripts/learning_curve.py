from TimeSeries_Helpers import *

# compute Learning Curve (how does the model perform with data from more and more users)

number_participants = list(range(1, 21))
results = [[] for i in range(20)]
for i in range(len(number_participants)):
    print(list(range(0, i+1)))
    config = AttrDict(
        merged_labels=False,
        threshold=80,
        interval_length=60,
        stride_train=5,
        stride_eval=10,
        context_length=0,
        batch_tfms=[TSStandardize()],     # irrelevant for MiniRocket
        batch_size=64,                    # irrelevant for MiniRocket
        train_ids=list(range(0, i+1)),
        valid_ids=[],
        test_ids=list(range(20, 30)),
        use_lvl1=True,
        use_lvl2=True,
        verbose=False,
        model="MiniRocket",
        n_estimators=40,
        oversampling=True,
        undersampling=False,
        features={"x-position": True, "y-position": True, "Distance Agent": True,
                  "Distance User": True, "AoI": True, "Dilation Left": True, "Dilation Right": True, "AI-position x": True, "AI-position y": True, "User-position x": True, "User-position y": True},
    )
    for j in range(5):
        model, run_res = train_miniRocket(config, "learningcurve", "curve")
        results[i].append(run_res["test_accuracy_accumulated"])

print("\n\n\nThe learning curve scores are:\n\n", results)
