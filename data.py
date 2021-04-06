import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from configs import DATA_DIR

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Data:
    _DAYS = 90

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
        return df

    @property
    def incoming(self):
        df = self.raw
        return df[df["game_time"] > datetime.now()]


if __name__ == "__main__":
    data = Data(alliance="中華職棒")
    df = data.incoming
    print(df)
