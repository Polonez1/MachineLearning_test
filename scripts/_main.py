import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List

# import load_data
# import machine_learning
#
app = FastAPI()


#
#
# COL_TO_TEST = [
#    "home_club_position",
#    "away_club_position",
#    "squad_size_x",
#    "average_age_x",
# ]
# df_full = load_data.load_df()
# df_transformed = machine_learning.transform_data(df_full)
# df, df_random = load_data.split_data_random_and_Xy(df_transformed)
# X_new_data = df_random[COL_TO_TEST]
#
#
class Score(BaseModel):
    home_club_position: float
    away_club_position: float
    squad_size_x: float
    average_age_x: float


# python -m uvicorn _main:app --reload
@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.joblib")


@app.post("/predict")
def predict(tip: Score):
    data = pd.DataFrame([dict(tip)])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}
