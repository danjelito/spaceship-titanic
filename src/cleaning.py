import re


def camel_to_snake(text):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()


def cleaning(df):

    # rename columns
    df = df.rename(columns=lambda c: camel_to_snake(c))
    df = df.rename(columns={"v_i_p": "vip", "v_r_deck": "vr_deck"})

    # change dtype
    dtypes = {
        "vip": "float",
        "cryo_sleep": "float",
    }
    df = df.astype(dtypes)

    return df
