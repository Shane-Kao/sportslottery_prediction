import os
from datetime import datetime, timedelta

from configs import DATA_DIR


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
    def data_path(self):
        return os.path.join(DATA_DIR, self.alliance)

    @property
    def files(self):
        files_ = os.listdir(self.data_path)
        files_ = sorted(files_, key=lambda x: datetime.strptime(x.split('.')[0], "%Y%m%d"), reverse=True)
        return files_


if __name__ == "__main__":
    data = Data(alliance="NBA")
    for file in data:
        print(file)
