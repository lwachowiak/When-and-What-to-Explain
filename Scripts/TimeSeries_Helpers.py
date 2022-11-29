from tsai.all import *
import numpy as np
from fastai.callback.wandb import *
import wandb
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def most_common(lst):
    '''
    Returns the most common element in a list
    '''
    data = Counter(lst)
    return max(lst, key=data.get)


def create_single_label_per_interval_with_context(df, interval_length, stride, features, context_length):
    '''
    Splits data into intervals of length interval_length + contextlength. The label is based only on the last frames of length interval_length
    :param df: dataframe containing the gaze data and labels
    :param interval_length: length of one of the resulting samples
    :param stride: by how many frames the next interval sample is moved. Non-overlaping if stride==interval_length
    :param features: list of features used
    :param context_length: length of the additional context
    :return: labeled interval data and original labels (not cut into intervals with majority-based label)
    '''
    labels = np.empty(0)
    values = []
    for i in range(0, len(df), stride):
        if i + interval_length + context_length <= len(df):
            interval_labels = list(
                df["Condition"][i+context_length:i+context_length+interval_length])
            majority_label = most_common(interval_labels)
            labels = np.append(labels, majority_label)
            # determine sample values (sample is 2d array of features, data)
            sample = []
            if features["AoI"]:
                sample.append(
                    list(df["Gaze_focus"][i:i+interval_length+context_length]))
            if features["Distance Agent"]:
                sample.append(
                    list(df["gaze_to_ai"][i:i+interval_length+context_length]))
            if features["Distance User"]:
                sample.append(
                    list(df["gaze_to_human"][i:i+interval_length+context_length]))
            if features["x-position"]:
                sample.append(
                    list(df["norm_pos_x"][i:i+interval_length+context_length]))
            if features["y-position"]:
                sample.append(
                    list(df["norm_pos_y"][i:i+interval_length+context_length]))
            if features["Dilation Left"]:
                sample.append(
                    list(df["diameter_left"][i:i+interval_length+context_length]))
            if features["Dilation Right"]:
                sample.append(
                    list(df["diameter_right"][i:i+interval_length+context_length]))
            if features["AI-position x"]:
                sample.append(
                    list(df["Agent x"][i:i+interval_length+context_length]))
            if features["AI-position y"]:
                sample.append(
                    list(df["Agent y"][i:i+interval_length+context_length]))
            if features["User-position x"]:
                sample.append(
                    list(df["Human x"][i:i+interval_length+context_length]))
            if features["User-position y"]:
                sample.append(
                    list(df["Human y"][i:i+interval_length+context_length]))
            values.append(sample)

    return values, labels, df["Condition"]


def recompute_treshold(df, thresh):
    df.loc[(df["gaze_to_human"] > thresh) & (df["gaze_to_ai"] > thresh),
           "Gaze_focus"] = 0  # "Env"
    df.loc[(df["gaze_to_human"] < thresh) & (df["gaze_to_ai"] > thresh),
           "Gaze_focus"] = 1  # "Human"
    df.loc[(df["gaze_to_human"] > thresh) & (df["gaze_to_ai"] < thresh),
           "Gaze_focus"] = 2  # "AI"
    df.loc[(df["gaze_to_human"] < thresh) & (df["gaze_to_ai"] < thresh) &
           (df["gaze_to_human"] < df["gaze_to_ai"]), "Gaze_focus"] = 1  # "Human"
    df.loc[(df["gaze_to_human"] < thresh) & (df["gaze_to_ai"] < thresh) &
           (df["gaze_to_human"] > df["gaze_to_ai"]), "Gaze_focus"] = 2  # AI"
    return df


def read_in_data(threshold, interval_length, stride_train, stride_eval, features, valid_ids, test_ids, context_length):
    """
    reads in merged csvs: applies recompute_treshold() and create_single_label_per_interval

    :param threshold: threshold for AoI analysis
    :param interval_length: length of one of the resulting samples
    :param stride_train: by how many frames the next interval sample is moved. Non-overlaping if stride==interval_length
    :param stride_eval: by how many frames the next interval sample is moved (used on val/test samples). Non-overlaping if stride==interval_length
    :param features: list of features used
    :param valid_ids: list of ids used for validation
    :param test_ids: list of ids used for testing
    :param context_length: length of the additional context

    :return: values and labels for lvl1 and lvl2 for all participants
    """
    path = "all-merged/"
    data = {}

    for i in range(1, 36):
        if i not in [11, 24, 26, 32, 34]:  # skip participants with no labels
            if i < 10:
                participant = "P0"+str(i)
            else:
                participant = "P"+str(i)
                data[participant] = {}
            # add dict entry
            data[participant] = {}

            data[participant]["000"] = recompute_treshold(
                pd.read_csv(path+participant+"_000.csv", sep=";"), threshold)
            data[participant]["001"] = recompute_treshold(
                pd.read_csv(path+participant+"_001.csv", sep=";"), threshold)

    lvl1_data = []
    lvl2_data = []

    for p_number, participant in enumerate(data):
        if p_number in valid_ids or p_number in test_ids:
            values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[participant]["000"], interval_length=interval_length, stride=stride_eval, features=features, context_length=context_length)
            lvl1_data.append((values, labels, raw_labels))
            values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[participant]["001"], interval_length=interval_length, stride=stride_eval, features=features, context_length=context_length)
            lvl2_data.append((values, labels, raw_labels))
        else:
            values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[participant]["000"], interval_length=interval_length, stride=stride_train, features=features, context_length=context_length)
            lvl1_data.append((values, labels, raw_labels))
            values, labels, raw_labels = create_single_label_per_interval_with_context(
                df=data[participant]["001"], interval_length=interval_length, stride=stride_train, features=features, context_length=context_length)
            lvl2_data.append((values, labels, raw_labels))
    return lvl1_data, lvl2_data, data


def dataPrep(threshold, interval_length, stride_train, stride_eval, train_ids, valid_ids, test_ids, use_lvl1, use_lvl2, merge_labels, batch_size, batch_tfms, features, context_length, oversampling, undersampling, verbose=False):
    '''
    :param threshold: threshold for AoI
    :param interval_length: how many frames make up one datapoint
    :param stride_train: how many frames are skipped between two datapoints (train)
    :param stride_eval: how many frames are skipped between two datapoints (eval)
    :param train_ids: list of participant ids for training
    :param valid_ids: list of participant ids for validation
    :param test_ids: list of participant ids for testing
    :param use_lvl1: boolean indicating if level 1 data is used
    :param use_lvl2: boolean indicating if level 2 data is used
    :param merge_labels: boolean indicating if confusion and error are merged
    :param batch_size: batch size for training
    :param batch_tfms: transformations
    :param features: list of timeseries features used
    :param context_length: length of context
    :param oversampling: boolean indicating if oversampling is used
    :param undersampling: boolean indicating if undersampling is used

    :return: dataloader and individual datasets/labels
    '''
    lvl1, lvl2, data = read_in_data(threshold=threshold, interval_length=interval_length,
                                    stride_train=stride_train, stride_eval=stride_eval, features=features, valid_ids=valid_ids, test_ids=test_ids, context_length=context_length)

    # prepare labels (1d array) and data (3D array) for TSAI
    X_train = np.empty(
        (0, sum(features.values()), interval_length+context_length), dtype=np.float64)
    X_val = np.empty((0, sum(features.values()),
                     interval_length+context_length), dtype=np.float64)
    X_test = np.empty((0, sum(features.values()),
                      interval_length+context_length), dtype=np.float64)
    y_train = np.empty(0)
    y_val = np.empty(0)
    y_test = np.empty(0)

    # for final eval after concatenation
    y_test_raw = []
    y_val_raw = []
    X_val_by_participant = []
    X_test_by_participant = []

    # merge data of participants based on ids
    for i in train_ids:
        if use_lvl1:
            y_train = np.append(y_train, lvl1[i][1])
            X_train = np.append(X_train, lvl1[i][0], axis=0)
        if use_lvl2:
            y_train = np.append(y_train, lvl2[i][1])
            X_train = np.append(X_train, lvl2[i][0], axis=0)
    for i in valid_ids:
        if use_lvl1:
            y_val = np.append(y_val, lvl1[i][1])
            X_val = np.append(X_val, lvl1[i][0], axis=0)
            y_val_raw.append(lvl1[i][2])
            X_p = np.empty((0, sum(features.values()),
                            interval_length+context_length), dtype=np.float64)
            X_p = np.append(X_p, lvl1[i][0], axis=0)
            X_val_by_participant.append(X_p)
        if use_lvl2:
            y_val = np.append(y_val, lvl2[i][1])
            X_val = np.append(X_val, lvl2[i][0], axis=0)
            y_val_raw.append(lvl2[i][2])
            X_p = np.empty((0, sum(features.values()),
                            interval_length+context_length), dtype=np.float64)
            X_p = np.append(X_p, lvl2[i][0], axis=0)
            X_val_by_participant.append(X_p)
    for i in test_ids:
        if use_lvl1:
            y_test = np.append(y_test, lvl1[i][1])
            X_test = np.append(X_test, lvl1[i][0], axis=0)
            y_test_raw.append(lvl1[i][2])
            X_p = np.empty((0, sum(features.values()),
                            interval_length+context_length), dtype=np.float64)
            X_p = np.append(X_p, lvl1[i][0], axis=0)
            X_test_by_participant.append(X_p)
        if use_lvl2:
            y_test = np.append(y_test, lvl2[i][1])
            X_test = np.append(X_test, lvl2[i][0], axis=0)
            y_test_raw.append(lvl2[i][2])
            X_p = np.empty((0, sum(features.values()),
                            interval_length+context_length), dtype=np.float64)
            X_p = np.append(X_p, lvl2[i][0], axis=0)
            X_test_by_participant.append(X_p)

    if oversampling:
        ros = RandomOverSampler(random_state=0, sampling_strategy={"Normal": Counter(y_train)[
                                "Normal"], "Confusion": Counter(y_train)["Confusion"], "Error": Counter(y_train)["Confusion"]})
        ros.fit_resample(X_train[:, :, 0], y_train)
        X_train = X_train[ros.sample_indices_]
        y_train = y_train[ros.sample_indices_]
    if undersampling:
        ros = RandomUnderSampler(random_state=0, sampling_strategy={"Normal": Counter(y_train)[
                                 "Confusion"], "Confusion": Counter(y_train)["Confusion"], "Error": Counter(y_train)["Error"]})
        ros.fit_resample(X_train[:, :, 0], y_train)
        X_train = X_train[ros.sample_indices_]
        y_train = y_train[ros.sample_indices_]

    if merge_labels:
        y_train[y_train == "Error"] = "Confusion"
        if len(y_val) > 0:
            y_val[y_val == "Error"] = "Confusion"
        if len(y_test) > 0:
            y_test[y_test == "Error"] = "Confusion"

    if verbose:
        print("Train Labels:", y_train.shape)
        print("Val Labels", y_val.shape)
        print("Test Labels:", y_test.shape)
        print("Train Data:", X_train.shape)
        print("Val Data:", X_val.shape)
        print("Test Data:", X_test.shape)
        if len(y_val) > 0:
            print("Normal Val Samples:", (y_val == "Normal").sum()/len(y_val))
            print("Confusion Val Samples:",
                  (y_val == "Confusion").sum()/len(y_val))
            print("Error Val Samples:", (y_val == "Error").sum()/len(y_val))
        if len(y_test) > 0:
            print("Normal Test Samples:", (y_test == "Normal").sum()/len(y_test))
            print("Confusion Test Samples:",
                  (y_test == "Confusion").sum()/len(y_test))
            print("Error Test Samples:", (y_test == "Error").sum()/len(y_test))

    X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])

    tfms = [None, TSClassification()]  # transforms for the data
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[
                                   batch_size, 128], batch_tfms=batch_tfms, num_workers=0)
    return dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant


def vsBaseline(input: Tensor, targs: Tensor):
    "Computes difference of achieved accuracy and baseline accuracy; majority class is hardcoded"
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    accuracy = (input == targs).float().mean()
    # compute baseline as the accuracy of always predicting the majority class
    majority_class = tensor(2)
    baseline_accuracy = (majority_class == targs).float().mean()
    return accuracy-baseline_accuracy


def vsBaseline_merged(input: Tensor, targs: Tensor):
    "Computes difference of achieved accuracy and baseline accuracy given only 2 labels; majority class is hardcoded"
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    accuracy = (input == targs).float().mean()
    # compute baseline as the accuracy of always predicting the majority class
    majority_class = tensor(1)
    baseline_accuracy = (majority_class == targs).float().mean()
    return accuracy-baseline_accuracy


def load_config_model(config, dls, cbs):
    if config.merged_labels:  # baseline metric has hardcoded majority class and needs to know if labels are merged
        metrics = [accuracy, F1Score(average="macro"), Precision(
            average="macro"), Recall(average="macro"), vsBaseline_merged]
    else:
        metrics = [accuracy, F1Score(average="macro"), Precision(
            average="macro"), Recall(average="macro"), vsBaseline]

    if config.model == "InceptionTime":
        model = InceptionTime(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "InceptionTimePlus":
        model = InceptionTimePlus(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "TST":
        model = TST(dls.vars, dls.c, dls.len,
                    dropout=config.dropout_TST, fc_dropout=config.fc_dropout_TST)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLossFlat())
        else:
            return Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(),  metrics=metrics, cbs=cbs)
    elif config.model == "XceptionTime":
        model = XceptionTime(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "ResNet":
        model = ResNet(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "xresnet1d34":
        model = xresnet1d34(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "ResCNN":
        model = ResCNN(dls.vars, dls.c)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "OmniScaleCNN":
        model = OmniScaleCNN(dls.vars, dls.c, dls.len)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "mWDN":
        model = mWDN(dls.vars, dls.c, dls.len)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "LSTM_FCN":
        model = LSTM_FCN(dls.vars, dls.c, dls.len,
                         fc_dropout=config.fc_dropout_LSTM_FCN, rnn_dropout=config.dropout_LSTM_FCN)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "LSTM":
        model = LSTM(dls.vars, dls.c, n_layers=3, bidirectional=True)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)
    elif config.model == "gMLP":
        model = gMLP(dls.vars, dls.c, dls.len)
        if config.focal_loss:
            return Learner(dls, model, metrics=metrics, cbs=cbs, loss_func=FocalLoss())
        else:
            return Learner(dls, model, metrics=metrics, cbs=cbs)


def evaluate_preds_against_raw(y_preds_per_participant, y_raw_per_participant, stride, context_length, interval_length, plotting):
    all_y_true_concatenated = []
    all_y_pred_concatenated = []
    for i, y_preds in enumerate(y_preds_per_participant):
        y_preds_concatenated = [y_preds[0]]*(interval_length-stride)
        y_raw = y_raw_per_participant[i]
        for j in range(1, len(y_preds)):
            y_preds_concatenated += ([y_preds[j]]*stride)
        # compute accuracy
        # make y_preds_concatenated as long as y_raw and fill the missing values with the "Normal"
        y_preds_concatenated += ["Normal"] * \
            (len(y_raw)-len(y_preds_concatenated))
        accuracy = accuracy_score(
            y_raw, y_preds_concatenated)  # old: y_raw[:len(y_preds_concatenated)]
        baseline = len(y_raw[y_raw == "Normal"])/len(y_raw)
        all_y_true_concatenated += list(y_raw)
        all_y_pred_concatenated += y_preds_concatenated
        if plotting:
            # plot y_raw and y_preds
            print("Accuracy: ", accuracy, "Baseline: ", baseline)
            plt.plot(y_raw[:len(y_preds_concatenated)], label="y_raw")
            plt.plot(y_preds_concatenated, label="y_preds")
            plt.legend()
            plt.show()
    accuracy = accuracy_score(all_y_true_concatenated, all_y_pred_concatenated)
    macro_f1 = f1_score(
        all_y_true_concatenated, all_y_pred_concatenated, average="macro")
    baseline = all_y_true_concatenated.count(
        "Normal") / len(all_y_true_concatenated)
    all_y_true_concatenated = [
        0 if x == "Confusion" else x for x in all_y_true_concatenated]
    all_y_true_concatenated = [
        1 if x == "Error" else x for x in all_y_true_concatenated]
    all_y_true_concatenated = [
        2 if x == "Normal" else x for x in all_y_true_concatenated]
    all_y_pred_concatenated = [
        0 if x == "Confusion" else x for x in all_y_pred_concatenated]
    all_y_pred_concatenated = [
        1 if x == "Error" else x for x in all_y_pred_concatenated]
    all_y_pred_concatenated = [
        2 if x == "Normal" else x for x in all_y_pred_concatenated]
    precision, recall, f1, support = precision_recall_fscore_support(
        all_y_true_concatenated, all_y_pred_concatenated, average=None)
    return accuracy, baseline, precision, recall, f1, support, macro_f1, all_y_true_concatenated, all_y_pred_concatenated


def evaluate(config, group, name, valid_preds, y_val, test_preds, y_test, y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {}
    if (len(config.valid_ids) > 0) and (config.valid_ids != config.test_ids):
        results["val_accuracy_accumulated"], results["val_baseline_accumulated"], results["val_precision_accumulated"], results["val_recall_accumulated"], results["val_f1_accumulated"], results["val_support_accumulated"], results["val_macroF1_accumulated"], results["all_y_true_concatenated"], results["all_y_pred_concatenated"] = evaluate_preds_against_raw(
            val_preds_per_participant, y_val_raw, stride=config["stride_eval"], context_length=config["context_length"], interval_length=config["interval_length"], plotting=False)
        results["val_vsBaseline_accumulated"] = results["val_accuracy_accumulated"] - \
            results["val_baseline_accumulated"]
        results["val_accuracy"] = accuracy_score(y_val, valid_preds)
        baseline = (y_val == "Normal").sum()/len(y_val)
        results["val_vsBaseline"] = results["val_accuracy"]-baseline
        cm_val = confusion_matrix(y_val, valid_preds)
        print(classification_report(y_val, valid_preds))
        results["val_precision"], results["val_recall"], results["val_fscore"], results["val_support"] = precision_recall_fscore_support(
            y_val, valid_preds, average=None)
        results["val_macroF1"] = f1_score(
            y_val, valid_preds, average="macro")

    # test performance
    if len(config.test_ids) > 0:
        results["test_accuracy_accumulated"], results["test_baseline_accumulated"], results["test_precision_accumulated"], results["test_recall_accumulated"], results["test_f1_accumulated"], results["test_support_accumulated"], results["test_macroF1_accumulated"],  results["all_y_true_concatenated"], results["all_y_pred_concatenated"] = evaluate_preds_against_raw(
            test_preds_per_participant, y_test_raw, stride=config["stride_eval"], context_length=config["context_length"], interval_length=config["interval_length"], plotting=False)
        results["test_vsBaseline_accumulated"] = results["test_accuracy_accumulated"] - \
            results["test_baseline_accumulated"]
        results["test_accuracy"] = accuracy_score(y_test, test_preds)
        cm_test = confusion_matrix(y_test, test_preds)
        results["test_cm"] = sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Confusion", "Error", "Normal"], yticklabels=["Confusion", "Error", "Normal"])
        baseline = (y_test == "Normal").sum()/len(y_test)
        results["test_vsBaseline"] = results["test_accuracy"]-baseline
        print(classification_report(y_test, test_preds))
        results["test_precision"], results["test_recall"], results["test_fscore"], results["test_support"] = precision_recall_fscore_support(
            y_test, test_preds, average=None)
        results["test_macroF1"] = f1_score(
            y_test, test_preds, average="macro")
        # wandb.log(results)

    return results


def train_miniRocket(config, group, name):
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling)

    # model and train
    model = MiniRocketVotingClassifier(n_estimators=config.n_estimators)
    model.fit(X_train, y_train)

    # evaluate
    if len(config.valid_ids) > 0:
        valid_preds = model.predict(X_val)
    else:
        valid_preds = None
    if len(config.test_ids) > 0:
        test_preds = model.predict(X_test)
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        test_preds_per_participant.append(model.predict(X_p))
    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        val_preds_per_participant.append(model.predict(X_p))

    results = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)

    return model, results


def train_fastAI(config, group, name):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # special case for final test
    if len(config.valid_ids) == 0:
        config.valid_ids = config.test_ids
    # prepare data
    dls, X_train, y_train, X_val, y_val, X_test, y_test, y_val_raw, y_test_raw, X_val_by_participant, X_test_by_participant = dataPrep(threshold=config.threshold,
                                                                                                                                       interval_length=config.interval_length,
                                                                                                                                       stride_train=config.stride_train,
                                                                                                                                       stride_eval=config.stride_eval,
                                                                                                                                       train_ids=config.train_ids,
                                                                                                                                       valid_ids=config.valid_ids,
                                                                                                                                       test_ids=config.test_ids,
                                                                                                                                       use_lvl1=config.use_lvl1,
                                                                                                                                       use_lvl2=config.use_lvl2,
                                                                                                                                       merge_labels=config.merged_labels,
                                                                                                                                       batch_size=config.batch_size,
                                                                                                                                       batch_tfms=config.batch_tfms,
                                                                                                                                       features=config.features,
                                                                                                                                       verbose=config.verbose,
                                                                                                                                       context_length=config.context_length,
                                                                                                                                       oversampling=config.oversampling,
                                                                                                                                       undersampling=config.undersampling)

    # model and train
    cbs = None
    learn = load_config_model(config=config, dls=dls, cbs=cbs)
    learn.fit_one_cycle(config.n_epoch, config.lr)

    if config.merged_labels:
        majority_class = tensor(1)
    else:
        majority_class = tensor(2)

    # evaluate
    if len(config.valid_ids) > 0 and config.valid_ids != config.test_ids:
        valid_probas, valid_targets = learn.get_X_preds(
            X_val, y_val, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
        valid_preds = [learn.dls.vocab[p]
                       for p in np.argmax(valid_probas, axis=1)]
    else:
        valid_preds = None

    if len(config.test_ids) > 0:
        test_probas, test_targets = learn.get_X_preds(
            X_test, y_test, with_decoded=False)
        test_preds = [learn.dls.vocab[p]
                      for p in np.argmax(test_probas, axis=1)]
    else:
        test_preds = None

    test_preds_per_participant = []
    for X_p in X_test_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        test_preds_per_participant.append(pred_X_p)

    val_preds_per_participant = []
    for X_p in X_val_by_participant:
        probas_X_p, _ = learn.get_X_preds(X_p, with_decoded=False)
        pred_X_p = [learn.dls.vocab[p] for p in np.argmax(probas_X_p, axis=1)]
        val_preds_per_participant.append(pred_X_p)

    results = evaluate(config, group, name, valid_preds, y_val, test_preds, y_test,
                       y_val_raw, y_test_raw, val_preds_per_participant, test_preds_per_participant)

    return learn, results


def cross_validate(val_fold_size, config, group, name):
    if 20 % val_fold_size != 0:
        raise ValueError("val_fold_size must be a divisor of 20")

    f1_scores = []
    vsBaselines = []
    f1_scores_accumulated = []
    vsBaselines_accumulated = []

    # iterate over folds
    for i in range(0, 20, val_fold_size):
        train_ids = list(range(0, 20))
        valid_ids = list(range(i, i+val_fold_size))
        for j in range(i, i+val_fold_size):
            train_ids.remove(j)
        print(train_ids, valid_ids)
        config.train_ids = train_ids
        config.valid_ids = valid_ids
        config.test_ids = []

        if config.model == "MiniRocket":
            model, results = train_miniRocket(
                config=config, group=group, name="_iteration"+str(i)+"_var"+name)
        else:
            model, results = train_fastAI(
                config=config, group=group, name="_iteration"+str(i)+"_var"+name)
        f1_scores.append(results["val_macroF1"])
        vsBaselines.append(results["val_vsBaseline"])
        f1_scores_accumulated.append(results["val_macroF1_accumulated"])
        vsBaselines_accumulated.append(results["val_vsBaseline_accumulated"])

    # final test with training on 1 to 20 and testing on 21 to 30
    config.train_ids = list(range(0, 20))
    config.valid_ids = []
    config.test_ids = list(range(20, 30))
    if config.model == "MiniRocket":
        model, test_results = train_miniRocket(
            config, group=group, name="test_set_performance")
    else:
        model, test_results = train_fastAI(
            config, group=group, name="test_set_performance")

    # print results
    print("Evaluated", len(f1_scores), "folds")
    print("Avg F1 Score", sum(f1_scores)/len(f1_scores))
    print("Avg vsBaseline", sum(vsBaselines)/len(vsBaselines))
    # print std devs
    print("Standard Deviation F1", np.std(f1_scores))
    print("Standard Deviation vsBaseline", np.std(vsBaselines))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # , settings=wandb.Settings(start_method="fork")):
    with wandb.init(project="WhenToExplain", config=config, group="summary-"+group, name=now+"_"+group+"_"+str(config)):
        wandb.log({"CV AVG Macro F1": sum(f1_scores)/len(f1_scores),
                  "CV AVG vsBaseline": sum(vsBaselines)/len(vsBaselines),
                   "CV AVG Macro F1 acc.": sum(f1_scores_accumulated)/len(f1_scores_accumulated),
                   "CV AVG vsBaseline acc.": sum(vsBaselines_accumulated)/len(vsBaselines_accumulated),
                   "CV STD Macro F1": np.std(f1_scores),
                   "CV STD vsBaseline": np.std(vsBaselines)})
        wandb.log({"ConfusionMatrix-Test": wandb.plot.confusion_matrix(probs=None,
                                                                       y_true=test_results["all_y_true_concatenated"], preds=test_results[
                                                                           "all_y_pred_concatenated"],
                                                                       class_names=["Confusion", "Error", "Normal"])})
        del test_results["all_y_true_concatenated"]
        del test_results["all_y_pred_concatenated"]
        wandb.log(test_results)
