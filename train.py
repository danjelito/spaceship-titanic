import warnings
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src import model_dispatcher, module, scoring, feature_selection
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train_fold, y_train_fold)
            pred = model.predict(x_val_fold)

        # scoring
        scores = scoring.return_score(y_val_fold, pred)
        print(
            f"Fold {fold: <1} | model {model_name: <10}:  acc {scores.acc: .5f}  f1 {scores.f1: .5f}"
        )
        model_scores[model_name]["acc"].append(scores.acc)
        model_scores[model_name]["f1"].append(scores.f1)

# print score
for model_name, metrics in model_scores.items():
    avg_acc = np.mean(metrics["acc"])
    avg_f1 = np.mean(metrics["f1"])
    print()
    print(f"Model: {model_name}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

print("Done!")
