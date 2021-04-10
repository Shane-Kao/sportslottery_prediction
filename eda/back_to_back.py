import os

from data import Data
from configs import DATA_DIR

for alliance in os.listdir(DATA_DIR):
    data = Data(alliance=alliance)
    df = data.history

    df["total_score"] = df["away_score"] + df["home_score"]
    df["diff"] = df["home_score"] - df["away_score"]
    print(data.alliance)
    print("total: ", df["total_score"].mean())
    print("diff: ", df["diff"].mean())
    print(df.groupby(
        ["is_back_to_back", ]
    ).agg({'total_score': 'mean', 'diff': 'mean', 'game_time': "count"}).rename(columns={'game_time': 'count'}) \
           .reset_index())
    print('==========================================================')