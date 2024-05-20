import pandas as pd
from src.cleaning import cleaning
from src.preprocessing import preprocessing
from src.feature_engineering import feature_engineering
from src.scaling_encoding import scaling_encoding


def load_dataset():
    
    df_train = pd.read_csv("input/train.csv")
    df_test = pd.read_csv("input/test.csv")
    target = "Transported"
    x_train = df_train.drop(columns=[target])
    x_test = df_test
    y_train = df_train.loc[:, [target]].astype(int)
    assert x_train.shape[1] == x_test.shape[1]
    return x_train, y_train, x_test


def data_preparation(x_train, x_test):

    x_train = x_train.copy()
    x_test = x_test.copy()
    assert (x_train.columns == x_test.columns).all()

    # cleaning
    x_train = cleaning(x_train)
    x_test = cleaning(x_test)
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
