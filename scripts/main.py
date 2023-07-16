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

from scripts import load_data
from scripts import machine_learning

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

df_full = load_data.load_df()
df_transformed = machine_learning.transform_data(df_full)
df, df_random = load_data.split_data_random_and_Xy(df_transformed)

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
    "gausian": machine_learning.find_best_gausian_parameters(
        X_train[X_TRAIN_COL], y_train, cv=2
    ),
    "rfc": machine_learning.find_random_forest_class_parameters(
        X[X_TRAIN_COL], y, cv=2
    ),
    "decisionTree": machine_learning.find_decision_tree_parameters(
        X_train[X_TRAIN_COL], y_train, cv=2
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
models_params_table = machine_learning.add_models_to_grid_table(models_params_table)

best_columns_dict = {}
for i in models_params_table[["model_name", "params", "model"]].itertuples(index=False):
    name = i[0]
    params = i[1]
    model_funkc = i[2]
    model = model_funkc(params=params)
    best_columns = machine_learning.find_best_columns_to_model(
        columns_to_test=COL_TO_TEST,
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
    )
    best_columns_dict[name] = {
        "model": model,
        "best_columns": best_columns,
        "params": params,
    }

X_new_data = df_random[COL_TO_TEST]

y_new_result = df_random["result"]


predict_new_data = machine_learning.predict_new_data(
    model=best_columns_dict["rfc"]["model"],
    X_new_data=X_new_data,
    y_new_result=y_new_result,
)
