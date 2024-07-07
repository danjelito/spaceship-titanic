import time
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src import feature_selection, model_dispatcher, module, scoring
from src.utils import print_df_in_chunks

# load df
x_train, y_train, x_test = module.load_dataset()

# training and validation loop
n_splits = 5
model_scores = defaultdict(lambda: defaultdict(list))
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):

    # data split
    x_train_fold = x_train.loc[train_idx]
    y_train_fold = y_train.loc[train_idx]
    x_val_fold = x_train.loc[val_idx]
    y_val_fold = y_train.loc[val_idx]

    # cleaning, preprocessing, feature eng, encoding and scaling
    x_train_fold, x_val_fold = module.data_preparation(x_train_fold, x_val_fold)

    # cast to float
    x_train_fold = x_train_fold.values.astype("float32")
    x_val_fold = x_val_fold.values.astype("float32")
    y_train_fold = y_train_fold.values.astype("float32").ravel()
    y_val_fold = y_val_fold.values.astype("float32").ravel()

    # feature selection
    mask = feature_selection.get_selected_features(x_train_fold, y_train_fold)
    x_train_fold = x_train_fold[:, mask]
    x_val_fold = x_val_fold[:, mask]

    for model_name, model in model_dispatcher.models.items():

        # fit model
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # training for nn need reshaping the y
            if model_name == "net":
                model.fit(x_train_fold, y_train_fold.reshape(-1, 1))
            else:
                model.fit(x_train_fold, y_train_fold)
            
            pred = model.predict(x_val_fold)
            end_time = time.time()
            time_taken = end_time - start_time

        # scoring
        scores = scoring.return_score(y_val_fold, pred)
        print(
            f"Fold {fold: <1} | model {model_name: <10}:  acc {scores.acc: .5f}  f1 {scores.f1: .5f}  time {time_taken: .1f}s"
        )
        model_scores[model_name]["acc"].append(scores.acc)
        model_scores[model_name]["f1"].append(scores.f1)

# print score
dfs = []
for name, metrics in model_scores.items():
    avg_acc = np.mean(metrics["acc"])
    avg_f1 = np.mean(metrics["f1"])
    df = pd.DataFrame(data={"model": [name], "acc": [avg_acc], "f1": [avg_f1]})
    dfs.append(df)

df = pd.concat(dfs).reset_index(drop=True)
df = df.assign(harmonic_mean=lambda df_: scoring.harmonic_mean(df_["acc"], df_["f1"]))
df = df.sort_values("harmonic_mean", ascending=False, ignore_index=True)

module.print_df_in_chunks(df, 10)

print("Done!")
