from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def create_group_id(df):
    df["group_id"] = df["passenger_id"].str.split("_").str[0].astype(int)
    return df


def create_travel_in_group(df):
    df["travel_in_group"] = df["group_id"].duplicated(keep=False).astype("float")
    return df


def create_deck(df):
    df["deck"] = df["cabin"].str.split("/").str[0].fillna("BLANK")
    return df


def create_room_number(df):
    df["room_number"] = df["cabin"].str.split("/").str[1].fillna("BLANK")
    return df


def create_side(df):
    df["side"] = df["cabin"].str.split("/").str[2].fillna("BLANK")
    return df


def create_is_name_blank(df):
    df["is_name_blank"] = (df["name"].astype("str").str.upper() == "BLANK").astype(
        "float"
    )
    return df


def feature_engineering(df):
    df = create_group_id(df)
    df = create_travel_in_group(df)
    df = create_deck(df)
    df = create_room_number(df)
    df = create_side(df)
    df = create_is_name_blank(df)
    return df
