from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline


import pandas as pd
import numpy as np

import load_data
import machine_learning
import joblib

# from scripts import machine_learning

X_TRAIN_COL = [
    "home_club_position",
    "away_club_position",
    "home_club_id",
    "away_club_id",
    "attendance",
    "squad_size_x",
    "average_age_x",
    "foreigners_percentage_x",
    "national_team_players_x",
]

MODEL_GRID_COL = [
    "model_name",
    "mean_fit_time",
    "std_fit_time",
    "mean_score_time",
    "std_score_time",
    "params",
    "mean_test_score",
    "std_test_score",
]

COL_TO_TEST = [
    "home_club_position",
    "away_club_position",
    "squad_size_x",
    "average_age_x",
]


def train():
    df_full = load_data.load_df()
    df_transformed = machine_learning.transform_data(df_full)
    # df, df_random = load_data.split_data_random_and_Xy(df_transformed)
    X, y = machine_learning.split_X_y(df_transformed)
    tts = train_test_split(X, y, stratify=y, test_size=0.3)
    X_train, X_test, y_train, y_test = tts

    params = {
        "simpleimputer__strategy": "mean",
        "randomforestclassifier__n_estimators": 20,
        "randomforestclassifier__max_depth": 10,
        "randomforestclassifier__criterion": "gini",
    }

    model = machine_learning.random_forest_class(params=params)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.joblib")


if "__main__" == __name__:
    train()
