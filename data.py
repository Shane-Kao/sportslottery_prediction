import os

from configs import DATA_DIR


class Data:
    def __init__(self, alliance):
        self.alliance = alliance

    @property
    def data_path(self):
        return os.path.join(DATA_DIR, self.alliance)

    @property
    def files(self):
        files_ = os.listdir(self.data_path)
        #TODO: sort by datetime
        return files_


if __name__ == "__main__":
    data = Data(alliance="NBA")
    print(data.data_path)
    print(data.files)