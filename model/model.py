# -*- coding: utf-8 -*-
__author__ = 'Shane_Kao'
from sklearn.model_selection import PredefinedSplit

from data import Data


class Model:
    TEST_SIZE = (0.1, 10, 100, )

    def __init__(self, alliance, book_maker, type_of_bet):
        self.alliance = alliance
        self.book_maker = book_maker
        self.type_of_bet = type_of_bet

    def _get_best(self):
        raw_data = Data(alliance=self.alliance)
        train_data = raw_data.get_train(book_maker=self.book_maker, type_of_bet=self.type_of_bet)
        target = train_data["{}_{}_result".format(self.book_maker, self.type_of_bet)].astype(int)
        test_size = int(train_data.shape[0] * self.TEST_SIZE[0])
        test_size = self.TEST_SIZE[1] if test_size < self.TEST_SIZE[1] else self.TEST_SIZE[2] if \
            test_size > self.TEST_SIZE[2] else test_size
        train_size = train_data.shape[0] - test_size
        setattr(self, "train_size_", train_size)
        setattr(self, "test_size_", test_size)
        test_fold = [-1 for _ in range(train_size)] + [0 for _ in range(test_size)]
        predefined_split = PredefinedSplit(test_fold)
        setattr(self, "predefined_split_", predefined_split)



if __name__ == '__main__':
    model = Model(alliance="NBA", book_maker="tw", type_of_bet="total")
    print(model.train_size_)
    print(model.test_size_)



