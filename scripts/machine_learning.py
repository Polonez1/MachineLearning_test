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

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler

import pandas as pd
import numpy as np


def transform_data(df: pd.DataFrame):
    df_copy = df.copy()

    cat_transformer = make_pipeline(
        OrdinalEncoder(categories="auto"),
    )

    num_transformer = make_pipeline(MinMaxScaler())

    preprocessor = ColumnTransformer(
        transformers=[
            ("result", cat_transformer, ["result"]),
            ("num", num_transformer, ["attendance"]),
        ]
    )
    transformed_values = preprocessor.fit_transform(df_copy)

    df_copy[["result", "attendance"]] = transformed_values

    return df_copy


def split_X_y(df: pd.DataFrame):
    X = df[
        [
            "home_club_position",
            "away_club_position",
            "attendance",
            "squad_size_x",
            "average_age_x",
            "foreigners_percentage_x",
            "national_team_players_x",
        ]
    ]

    y = df["result"]

    return X, y


def SimpleImputer_transform(df: pd.DataFrame):
    simple_imputer = SimpleImputer(strategy="mean")
    transformed_values = simple_imputer.fit_transform(df)
    df_transformed_filled = pd.DataFrame(transformed_values, columns=df.columns)
    return df_transformed_filled


def test_linear_model(
    X,
    y,
    cv=5,
    polynomialfeatures__degree: list = [2, 3, 4],
    linearregression__n_jobs: list = [1, 5, 10],
):
    model = make_pipeline(SimpleImputer(), PolynomialFeatures(), LinearRegression())
    params = {
        "simpleimputer__strategy": ["mean", "median"],
        "polynomialfeatures__degree": polynomialfeatures__degree,
        "linearregression__n_jobs": linearregression__n_jobs,
    }

    grid = GridSearchCV(model, param_grid=params, cv=cv)
    grid.fit(X, y)
    best_parameter = grid.best_params_

    return best_parameter


def test_gausian_model(
    X,
    y,
    cv=5,
):
    model = make_pipeline(SimpleImputer(), GaussianNB())
    params = {
        "simpleimputer__strategy": ["mean", "median"],
    }

    grid = GridSearchCV(model, param_grid=params, cv=cv)
    grid.fit(X, y)
    best_parameter = grid.best_params_

    return best_parameter


def create_linear_model(params: dict):
    best_model = make_pipeline(
        SimpleImputer(strategy=params["simpleimputer__strategy"]),
        PolynomialFeatures(degree=params["polynomialfeatures__degree"]),
        LinearRegression(n_jobs=params["linearregression__n_jobs"]),
    )

    return best_model


def create_gausian_model(params: dict):
    best_model = make_pipeline(
        SimpleImputer(strategy=params["simpleimputer__strategy"]), GaussianNB()
    )

    return best_model
