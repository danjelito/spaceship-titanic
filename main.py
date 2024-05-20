import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import (
    cleaning,
    feature_engineering,
    model_dispatcher,
    scaling_encoding,
    scoring,
)
from src.feature_engineering import feature_engineering
from src.preprocessing import preprocessing
from src.scaling_encoding import scaling_encoding
from src.utils import print_df_in_chunks


# load df
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")
target = "Transported"
x_train = df_train.drop(columns=[target])
x_test = df_test
y_train = df_train.loc[:, [target]].astype(int)
assert x_train.shape[1] == x_test.shape[1]


def data_preparation(x_train, x_test):

    x_train = x_train.copy()
    x_test = x_test.copy()
    assert (x_train.columns == x_test.columns).all()

    # cleaning
    x_train = cleaning.clean_data(x_train)
    x_test = cleaning.clean_data(x_test)
    assert (x_train.columns == x_test.columns).all()

    # preprocessing
    preprocessing.fit(x_train)
    column_names = [col.split("__")[1] for col in preprocessing.get_feature_names_out()]
    x_train = pd.DataFrame(preprocessing.transform(x_train), columns=column_names)
    x_test = pd.DataFrame(preprocessing.transform(x_test), columns=column_names)
    assert x_train.isna().sum().sum() == 0
    assert x_test.isna().sum().sum() == 0
    assert (x_train.columns == x_test.columns).all()

    # feature engineering
    x_train = feature_engineering(x_train)
    x_test = feature_engineering(x_test)
    assert x_train.isna().sum().sum() == 0
    assert x_test.isna().sum().sum() == 0
    assert (x_train.columns == x_test.columns).all()

    # encoding and scaling
    scaling_encoding.fit(x_train)
    column_names = [
        col.split("__")[1] for col in scaling_encoding.get_feature_names_out()
    ]
    x_train = pd.DataFrame(scaling_encoding.transform(x_train), columns=column_names)
    x_test = pd.DataFrame(scaling_encoding.transform(x_test), columns=column_names)
    assert x_train.isna().sum().sum() == 0
    assert x_test.isna().sum().sum() == 0
    assert (x_train.columns == x_test.columns).all()

    return x_train, x_test


# training and validation loop
n_splits = 5
model_scores = defaultdict(lambda: defaultdict(list))

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):

    # prepare data
    x_train_fold = x_train.loc[train_idx]
    y_train_fold = y_train.loc[train_idx]
    x_val_fold = x_train.loc[val_idx]
    y_val_fold = y_train.loc[val_idx]
    x_train_fold, x_val_fold = data_preparation(x_train_fold, x_val_fold)
    x_train_fold = x_train_fold.values.astype("float32")
    x_val_fold = x_val_fold.values.astype("float32")
    y_train_fold = y_train_fold.values.astype("float32")
    y_val_fold = y_val_fold.values.astype("float32")

    for name, model in model_dispatcher.models.items():

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train_fold, y_train_fold)
            pred = model.predict(x_val_fold)

        scores = scoring.return_score(y_val_fold, pred)
        print(
            f"Fold {fold: <1} | model {name: <6}:  acc {scores.acc: .5f}  f1 {scores.f1: .5f}"
        )
        model_scores[name]["acc"].append(scores.acc)
        model_scores[name]["f1"].append(scores.f1)

for name, metrics in model_scores.items():
    avg_acc = np.mean(metrics["acc"])
    avg_f1 = np.mean(metrics["f1"])
    print()
    print(f"Model: {name}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}\n")

print("Done!")
