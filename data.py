import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from configs import DATA_DIR

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Data:
    _DAYS = 90
    _BASE_COLUMNS = ["game_time", "away_score", "home_score", "away_team", "home_team",
                     "away_team_is_back_to_back", "home_team_is_back_to_back"]
    _TW_DIFF_COLUMNS = ['tw_diff', 'tw_diff_away_odds', 'tw_diff_home_odds', 'tw_diff_home_count', 'tw_away_odds',
                        'tw_home_odds', 'tw_home_count']
    _TW_TOTAL_COLUMNS = ['tw_total', 'tw_under_odds', 'tw_over_odds', 'tw_over_count']
    _OVERSEA_DIFF_COLUMNS = ['oversea_diff', 'oversea_diff_home_count']
    _OVERSEA_TOTAL_COLUMNS = ['oversea_total', 'oversea_over_count']

    def __init__(self, alliance):
        self.alliance = alliance

    def __iter__(self):
        current_date = datetime.now()
        for i in self.files:
            file_date = datetime.strptime(i.split('.')[0], "%Y%m%d")
            date_diff = current_date - file_date
            if date_diff > timedelta(days=self._DAYS):
                break
            else:
                yield os.path.join(DATA_DIR, self.alliance, i)
                current_date = file_date

    @property
    def path(self):
        return os.path.join(DATA_DIR, self.alliance)

    @property
    def files(self):
        files_ = os.listdir(self.path)
        files_ = sorted(files_, key=lambda x: datetime.strptime(x.split('.')[0], "%Y%m%d"), reverse=True)
        return files_

    @property
    def raw(self):
        data_ = []
        for file in self:
            date_ = os.path.splitext(Path(file).name)[0]
            df_ = pd.read_csv(file, index_col=None, header=0)
            df_["game_time"] = df_["game_time"].apply(
                lambda x: datetime.strptime(date_ + x, "%Y%m%d%p %I:%M")
            )
            data_.append(df_)
        df = pd.concat(data_, axis=0, ignore_index=True)
        df = df.sort_values(by="game_time", ignore_index=True)
        df = self._get_back_to_back(df)
        return df

    @property
    def incoming(self):
        df = self.raw
        return df[df["game_time"] > datetime.now()]

    @property
    def history(self):
        df = self.raw
        df = df[df['away_score'].notnull() & df['home_score'].notnull()]
        return df

    def get_train(self, book_maker, type_of_bet):
        assert book_maker == "tw" or book_maker == "oversea"
        assert type_of_bet == "total" or type_of_bet == "diff"
        extra_cols = getattr(self, '_'.join(["", book_maker.upper(), type_of_bet.upper(), "COLUMNS"]))
        df = self.history[self._BASE_COLUMNS + extra_cols]
        df = df[df[extra_cols[0]].notnull()]
        return df

    @staticmethod
    def _get_back_to_back(df):
        last_game_time_dict = {}
        for idx, row in df.iterrows():
            game_time = row.game_time.date()
            away_team = row.away_team
            home_team = row.home_team
            away_team_last_game_time = last_game_time_dict.get(away_team)
            home_team_last_game_time = last_game_time_dict.get(home_team)
            away_team_is_back_to_back = False if away_team_last_game_time is None else True if \
                (game_time - away_team_last_game_time).days < 2 else False
            home_team_is_back_to_back = False if home_team_last_game_time is None else True if \
                (game_time - home_team_last_game_time).days < 2 else False
            last_game_time_dict[away_team] = game_time
            last_game_time_dict[home_team] = game_time
            df.loc[idx, "away_team_is_back_to_back"] = away_team_is_back_to_back
            df.loc[idx, "home_team_is_back_to_back"] = home_team_is_back_to_back
        return df


if __name__ == "__main__":
    data = Data(alliance="NBA")
    print(data.incoming[["game_time", "away_team", "home_team",
                         "away_team_is_back_to_back", "home_team_is_back_to_back"]])
    # df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    # print(df1)
    # df2 = data.get_train(book_maker="tw", type_of_bet="total")
    # print(df2)
    # df3 = data.get_train(book_maker="oversea", type_of_bet="diff")
    # print(df3)
    # df4 = data.get_train(book_maker="oversea", type_of_bet="total")
    # print(df4)
