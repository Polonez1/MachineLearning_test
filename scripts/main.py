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


df = load_data.load_df()
df_transformed = machine_learning.transform_data(df)


X, y = machine_learning.split_X_y(df_transformed)
tts = train_test_split(X, y, stratify=y, test_size=0.3)
X_train, X_test, y_train, y_test = tts

models_list = {
    machine_learning.test_linear_model: machine_learning.create_linear_model,
    machine_learning.test_gausian_model: machine_learning.create_gausian_model,
}


for model_test in models_list.items():
    params = model_test[0](
        X[
            [
                "home_club_position",
                "away_club_position",
            ]
        ],
        y,
        cv=2,
    )

    model = model_test[1](params=params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(model, score)
