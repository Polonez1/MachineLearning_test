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

    preprocessor = ColumnTransformer(
        transformers=[
            ("result", cat_transformer, ["result"]),
            ("attendance", cat_transformer, ["result"]),
        ]
    )

    transformed_values = preprocessor.fit_transform(df_copy)
    df_copy[["result", "attendance"]] = transformed_values

    # encoded_result = transformed_values[:, 0]
    #
    # categories_result = (
    #    preprocessor.named_transformers_["result"]
    #    .named_steps["ordinalencoder"]
    #    .categories_[0]
    # )
    # decoded_result = [categories_result[int(val)] for val in encoded_result]
    #
    # mapping = {
    #    label: encoded for label, encoded in zip(categories_result, encoded_result)
    # }

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


def find_best_linear_parameters(
    X,
    y,
    X_test,
    y_test,
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
    grid.cv_results_

    return grid.cv_results_


def find_best_gausian_parameters(X, y, cv=5):
    model = make_pipeline(SimpleImputer(), GaussianNB())
    params = {
        "simpleimputer__strategy": ["mean", "median"],
        "gaussiannb__priors": [
            None,
            [0.45, 0.3, 0.25],
        ],
        "gaussiannb__var_smoothing": [1e-9, 1e-6],
    }

    grid = GridSearchCV(model, param_grid=params, cv=cv)
    grid.fit(X, y)
    grid.cv_results_

    return grid.cv_results_


def find_random_forest_class_parameters(X, y, cv=5):
    model = make_pipeline(SimpleImputer(), RandomForestClassifier())
    params = {
        "simpleimputer__strategy": ["mean", "median"],
        "randomforestclassifier__n_estimators": [10, 25, 50],
        "randomforestclassifier__max_depth": [2, 5, 10],
        "randomforestclassifier__criterion": ["gini"],
    }
    grid = GridSearchCV(model, param_grid=params, cv=cv)
    grid.fit(X, y)
    grid.cv_results_

    return grid.cv_results_


def create_linear_model(params: dict):
    best_model = make_pipeline(
        SimpleImputer(strategy=params["simpleimputer__strategy"]),
        PolynomialFeatures(degree=params["polynomialfeatures__degree"]),
        LinearRegression(n_jobs=params["linearregression__n_jobs"]),
    )

    return best_model


def gaussian_model(params: dict):
    model = make_pipeline(
        SimpleImputer(strategy=params["simpleimputer__strategy"]),
        GaussianNB(
            priors=params["gaussiannb__priors"],
            var_smoothing=params["gaussiannb__var_smoothing"],
        ),
    )

    return model


def random_forest_class(params: dict):
    model = make_pipeline(
        SimpleImputer(strategy=params["simpleimputer__strategy"]),
        RandomForestClassifier(
            n_estimators=params["randomforestclassifier__n_estimators"],
            max_depth=params["randomforestclassifier__max_depth"],
            criterion=params["randomforestclassifier__criterion"],
        ),
    )

    return model


def get_params(df: pd.DataFrame, model_name: str) -> dict:
    model_row = df.loc[df["model_name"] == model_name]
    params = model_row["params"].values[0]

    return params


def create_prediction_table(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)

    train_results = pd.DataFrame({"y_train": y_train, "y_train_pred": y_train_pred})

    train_results = pd.concat([train_results, X_train], axis=1)

    return train_results
