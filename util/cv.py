import numpy as np
from sklearn.model_selection import KFold


def get_cv_ids(dataset_config):
    ids = open(dataset_config["train"]).read().split("\n")
    ids += open(dataset_config["val"]).read().split("\n")
    return np.array(ids)


def get_52_cv_ids(ids):
    cv_ids = []
    for i in range(5):
        kfold = KFold(n_splits=2, shuffle=True, random_state=i)
        for train_idx, val_idx in kfold.split(ids):
            cv_ids.append((train_idx, val_idx))

    return cv_ids
