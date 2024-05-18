import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from src import cleaning, feature_engineering
from src.preprocessing import preprocessing, column_order
from src.feature_engineering import feature_engineering
from src.utils import print_df_in_chunks
from tabulate import tabulate


# load df
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

# create x and y
target = "Transported"
x_train = df_train.drop(columns=[target])
x_test = df_test
y_train = df_train.loc[:, target].values.astype(int)
assert x_train.shape[1] == x_test.shape[1]

# cleaning
x_train = cleaning.clean_data(x_train)
x_test = cleaning.clean_data(x_test)

# preprocessing
preprocessing.fit(x_train)
x_train = pd.DataFrame(preprocessing.transform(x_train), columns=column_order)
x_test = pd.DataFrame(preprocessing.transform(x_test), columns=column_order)
assert x_train.isna().sum().sum() == 0
assert x_test.isna().sum().sum() == 0

# feature engineering
x_train = feature_engineering(x_train)
x_test = feature_engineering(x_test)
assert x_train.isna().sum().sum() == 0
assert x_test.isna().sum().sum() == 0

# encoding and scaling


print_df_in_chunks(x_train, 5)
print("Done")

# # training and validation loop
# n_splits = 5
# avg_acc = 0
# avg_f1 = 0
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
# for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
#     # initialize model, imputer and scaler
#     xgb = XGBClassifier()
#     imputer = KNNImputer()
#     scaler = StandardScaler()
#     # prepare data
#     x_train_fold = x_train[train_idx]
#     y_train_fold = y_train[train_idx]
#     x_val_fold = x_train[val_idx]
#     y_val_fold = y_train[val_idx]
#     # cleaning and preprocessing
#     x_train_fold = module.pipeline(x_train_fold)
#     x_train_fold = module.pipeline(x_train_fold)

#     xgb.fit(x_train_fold, y_train_fold)
#     pred = xgb.predict(x_val_fold)
#     scores = module.return_score(y_val_fold, pred)
#     print(f"Fold {fold: <2}:  acc {scores.acc: .5f}  f1 {scores.f1: .5f}")
#     avg_acc += scores.acc
#     avg_f1 += scores.f1

# print(f"Average scores:  acc {avg_acc/n_splits: .5f}  f1 {avg_f1/n_splits: .5f}")

# print("Done!")
