# -*- coding: utf-8 -*-
__author__ = 'Shane_Kao'
import os
from datetime import datetime

import dill
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import make_scorer, precision_score

from data import Data
from feature_union import features
from model.params import get_base_params
from configs import MODEL_DIR


class Model:
    TEST_SIZE = (0.1, 10, 100, )

    PIPELINE = Pipeline([
        ('features', features),
        ('feature_selection', SelectPercentile()),
        ('clf', None)
    ])

    SCORING = make_scorer(precision_score, average="micro")

    N_ITER = 500

    N_JOBS = 4

    _MIN_COUNT = 20

    def __init__(self, alliance, book_maker, type_of_bet):
        self.alliance = alliance
        self.book_maker = book_maker
        self.type_of_bet = type_of_bet

    def _get_best(self):
        raw_data = Data(alliance=self.alliance)
        train_data = raw_data.get_train(book_maker=self.book_maker, type_of_bet=self.type_of_bet)
        if train_data.empty or train_data.shape[0] < self._MIN_COUNT:
            return {"status": False, "msg": " The data is insufficient.", }
        try:
            setattr(self, "start_date_", str(train_data.iloc[0, 0].date()))
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
            params = get_base_params(self.book_maker, self.type_of_bet)
            rand_search_cv = RandomizedSearchCV(
                estimator=self.PIPELINE,
                param_distributions=params,
                n_jobs=self.N_JOBS,
                verbose=0,
                cv=self.predefined_split_,
                scoring=self.SCORING,
                refit=False,
                n_iter=self.N_ITER,
            )
            rand_search_cv.fit(train_data, target)
            setattr(self, "best_score_", rand_search_cv.best_score_)
            setattr(self, "best_params_", rand_search_cv.best_params_)
            return {"status": True, "train_data": train_data, "target": target}
        except Exception as e:
            return {"status": False, "msg": str(e), }

    def train(self):
        result_dict = self._get_best()
        status = result_dict["status"]
        if status:
            train_data = result_dict["train_data"]
            target = result_dict["target"]
            train_index, test_index = list(self.predefined_split_.split())[0]
            X_train, X_test = train_data.iloc[train_index, :], train_data.iloc[test_index, :]
            y_train, y_test = target.iloc[train_index,], target.iloc[test_index,]
            self.PIPELINE.set_params(**self.best_params_)
            self.PIPELINE.fit(X=X_train, y=y_train)
            y_pred = self.PIPELINE.predict(X_test)
            p_micro = precision_score(y_true=y_test, y_pred=y_pred, average="micro")
            p0 = precision_score(y_true=y_test, y_pred=y_pred, pos_label=0)
            p1 = precision_score(y_true=y_test, y_pred=y_pred, pos_label=1)
            X_test.loc[:, "pred_"] = y_pred
            X_test.loc[:, "result"] = X_test.apply(lambda x: "準" if x.pred_ == x["{}_{}_result".
                                            format(self.book_maker, self.type_of_bet)] else "冏",
                                            axis=1)
            if self.type_of_bet == "total":
                X_test.loc[:, "pred"] = ["大" if i else "小" for i in y_pred]
            else:
                X_test.loc[:, "pred"] = X_test.apply(lambda x: "主" if x.pred_ else "客", axis=1)
            test_results = X_test[["game_time", "away_team", "home_team",
                                   "{}_{}".format(self.book_maker, self.type_of_bet),
                                   "pred", "result"]]
            setattr(self, "test_results_", test_results)
            setattr(self, "p_micro_", p_micro)
            setattr(self, "p0_", p0)
            setattr(self, "p1_", p1)
            setattr(self, "create_time_", str(datetime.now()))
            return True
        else:
            return False

    def save(self):
        model_path = os.path.join(MODEL_DIR, "{}_{}_{}".format(self.alliance, self.book_maker,
                                                               self.type_of_bet))
        result_dict = {
            "start_date": self.start_date_,
            "create_time": self.create_time_,
            "best_score": self.best_score_,
            "p_micro": self.p_micro_,
            "p0": self.p0_,
            "p1": self.p1_,
            "test_results": list(self.test_results_.T.to_dict().values()),
            "model": self.PIPELINE
        }
        dill.dump(result_dict, open(model_path, "wb"))
        return None


if __name__ == '__main__':
    from model._model import Model
    from configs import DATA_DIR

    alliances = os.listdir(DATA_DIR)
    book_makers = ['tw', 'oversea']
    type_of_bets = ['diff', 'total']

    for alliance in alliances:
        for book_maker in book_makers:
            for type_of_bet in type_of_bets:
                model = Model(
                    alliance=alliance,
                    book_maker=book_maker,
                    type_of_bet=type_of_bet
                )
                result = model.train()
                if result:
                    model.save()


