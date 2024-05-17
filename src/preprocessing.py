from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class BoolImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.imputer = IterativeImputer()

    def fit(self, x, y=None):
        x = x.copy()
        x = x.astype("float")
        self.imputer.fit(x)
        return self

    def transform(self, x, y=None):
        x = x.copy()
        x = x.astype("float")
        x = self.imputer.transform(x)
        x = x.round(0)
        return x

    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)


dtypes = {
    "str": [
        "passenger_id",
        "home_planet",
        "cryo_sleep",
        "cabin",
        "destination",
        "name",
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
        "vip",
    ],
}
str_pipeline = Pipeline(
    [("impute", SimpleImputer(strategy="constant", fill_value="BLANK"))]
)
float_pipeline = Pipeline([("impute", IterativeImputer())])
bool_pipeline = Pipeline([("impute", BoolImputer())])
preprocessing = ColumnTransformer(
    [
        ("str", str_pipeline, dtypes["str"]),
        ("float", float_pipeline, dtypes["float"]),
        ("bool", bool_pipeline, dtypes["bool"]),
    ]
)
