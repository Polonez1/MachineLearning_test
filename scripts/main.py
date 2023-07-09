from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler

import pandas as pd
import numpy as np

import load_data
import machine_learning

X_TRAIN_COL = [
    "home_club_position",
    "away_club_position",
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


df = load_data.load_df()
df_transformed = machine_learning.transform_data(df)


X, y = machine_learning.split_X_y(df_transformed)
tts = train_test_split(X, y, stratify=y, test_size=0.3)
X_train, X_test, y_train, y_test = tts


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

models_dict = {
    "gausian": machine_learning.find_best_gausian_parameters(X[X_TRAIN_COL], y, cv=5),
    "rfc": machine_learning.find_random_forest_class_parameters(
        X[X_TRAIN_COL], y, cv=5
    ),
}


grid_df = []
for model in models_dict.items():
    model_table = model[1]
    model_grid = pd.DataFrame(model_table)
    best_model_df = model_grid.loc[model_grid["rank_test_score"] == 1].head(1)
    model_name = model[0]
    best_model_df["model_name"] = model_name
    grid_df.append(best_model_df)

models_params_table = pd.concat(grid_df)[MODEL_GRID_COL]


params = machine_learning.get_params(models_params_table, "rfc")
model = machine_learning.random_forest_class(params=params)


prediction_table = machine_learning.create_prediction_table(
    model=model, X_train=X_train, y_train=y_train
)
