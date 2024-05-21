import re
import numpy as np


def camel_to_snake(text):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()


def cleaning(df):

    # rename columns
    df = df.rename(columns=lambda c: camel_to_snake(c))
    df = df.rename(columns={"v_i_p": "vip", "v_r_deck": "vr_deck"})

    # if cyro sleep is true, then spending will be 0
    def update_columns(df, columns):
        for column in columns:
            df[column] = np.where(
                (df["cryo_sleep"] == True) & (df[column].isna()), 0, df[column]
            )
        return df

    cols = [
        "room_service",
        "food_court",
        "shopping_mall",
        "spa",
        "vr_deck",
    ]
    df = update_columns(df, cols)

    # change dtype
    dtypes = {
        "vip": "float",
        "cryo_sleep": "float",
    }
    df = df.astype(dtypes)

    return df
