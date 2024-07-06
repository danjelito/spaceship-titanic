import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

dtypes = {
    "str": [
        "home_planet",
        "destination",
        "deck",
        "group_id",
        "side",
        "room_number",
    ],
    "float": [
        "age",
        "room_service",
        "food_court",
        "shopping_mall",
        "spa",
        "vr_deck",
    ],
    "bool": [
        "cryo_sleep",
        "vip",
        "travel_in_group",
        "is_name_blank",
    ],
    "drop": [
        "passenger_id",
        "cabin",
        "name",
    ],
}
str_pipeline = Pipeline([
    # onehot encode
    ("encode", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    # scale
    ("scale", StandardScaler()),
])
float_pipeline = Pipeline([
    # scale
    ("scale", StandardScaler()),
])
bool_pipeline = Pipeline([
    # scale
    ("scale", StandardScaler()),
])
# pipeline for all features
scaling_encoding = ColumnTransformer(
    [
        ("str", str_pipeline, dtypes["str"]),
        ("float", float_pipeline, dtypes["float"]),
        ("bool", bool_pipeline, dtypes["bool"]),
    ],
    remainder="drop",
)