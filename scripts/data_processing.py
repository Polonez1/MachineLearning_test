import pandas as pd


def eval_xG_xA(df: pd.DataFrame):
    df = pd.read_csv("./data/Premier_League_players.csv")
    df["xG"] = df["xG"].map(lambda x: eval(x))
    df["xA"] = df["xA"].map(lambda x: eval(x))

    return df
