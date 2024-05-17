import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

dtypes = {
    "str": [
    "home_planet",
    "destination",
    ],
    "float": [
    "age",
    ],
    "bool": [
    "cryo_sleep",
    ],
}


[
    "passenger_id",
    "vip",
    "cabin",
    "room_service",
    "food_court",
    "shopping_mall",
    "spa",
    "vr_deck",
    "name",
    "group_id",
    "travel_in_group",
    "deck",
    "room_number",
    "side",
    "is_name_blank",
]
