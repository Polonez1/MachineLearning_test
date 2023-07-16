from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import pandas as pd
import numpy as np

GAMES_COL = [
    "home_club_id",
    "away_club_id",
    "home_club_position",
    "away_club_position",
    "attendance",
    "home_club_name",
    "away_club_name",
    "result",
]

CLUB_COL = [
    "club_id",
    "squad_size",
    "average_age",
    "foreigners_percentage",
    "national_team_players",
]

FINAL_COL = [
    "home_club_position",
    "away_club_position",
    "home_club_id",
    "away_club_id",
    "attendance",
    "squad_size_x",
    "average_age_x",
    "foreigners_percentage_x",
    "national_team_players_x",
    "result",
]


def download_data():
    api = KaggleApi()
    api.authenticate()
    dataset_slug = "davidcariboo/player-scores"
    api.dataset_download_files(dataset_slug, path="./data/")


def unzip_data():
    zip_file_path = "./data/player-scores.zip"
    destination_folder = "./data/"

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(destination_folder)


def load_df():
    games = pd.read_csv("./data/games.csv")
    games = games.assign(
        result=lambda x: np.where(
            x["home_club_goals"] > x["away_club_goals"],
            "home_win",
            np.where(x["home_club_goals"] < x["away_club_goals"], "home_loss", "draw"),
        )
    )
    games = games[GAMES_COL]

    clubs = pd.read_csv("./data/clubs.csv")
    clubs = clubs[CLUB_COL]

    df = pd.merge(games, clubs, how="left", left_on="home_club_id", right_on="club_id")
    df = pd.merge(df, clubs, how="left", left_on="away_club_id", right_on="club_id")

    df = df[FINAL_COL]

    return df


def split_data_random_and_Xy(df: pd.DataFrame) -> pd.DataFrame:
    random_df = df.sample(n=400)
    df = df.drop(random_df.index)

    return df, random_df
