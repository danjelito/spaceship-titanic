import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


dtypes = {
    "str": [
        "passenger_id", "home_planet", "cabin", "destination", "name",
    ],
    "float": [
        "age", "room_service", "food_court", "shopping_mall", "spa", "vr_deck",
    ],
    "bool": [
        "cryo_sleep", "vip",
    ],
}
str_pipeline = Pipeline([
    # impute missing values
    ("impute", SimpleImputer(strategy="constant", fill_value="BLANK"))
])
float_pipeline = Pipeline([
    # impute missing values
    ("impute", IterativeImputer())
])
bool_pipeline = Pipeline([
    # impute missing values
    ("impute", IterativeImputer()),
    # make the range (0, 1) after imputation
    ("scale", MinMaxScaler(feature_range=(0, 1))),
    # round to make it 0 or 1 again (because boolean)
    ("round", FunctionTransformer(np.round, feature_names_out="one-to-one")),
])
# pipeline for all features
preprocessing = ColumnTransformer(
    [
        ("str", str_pipeline, dtypes["str"]),
        ("float", float_pipeline, dtypes["float"]),
        ("bool", bool_pipeline, dtypes["bool"]),
    ],
    remainder="passthrough",
)
# the order of resulting df will be based on the order in columntransformer
column_order = [col for cols in dtypes.values() for col in cols]
