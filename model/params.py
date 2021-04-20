# -*- coding: utf-8 -*-
__author__ = 'Shane_Kao'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier


def get_base_params(book_maker, type_of_bet):
    param = {
            'feature_selection__percentile': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                                              65, 70, 75, 80, 85, 90, 95],
            'features__feature_union__strk_encoder__kw_args': (
                {"book_maker": book_maker, "type_of_bet": type_of_bet},
            ),
            'features__feature_union__betting_encoder__kw_args': (
                {"book_maker": book_maker, "type_of_bet": type_of_bet},
            ),
            'features__feature_union__betting_count_encoder__kw_args': (
                {"book_maker": book_maker, "type_of_bet": type_of_bet},
            ),
            'features__feature_union__stats_encoder__kw_args': (
                {"type_of_bet": type_of_bet},
            ),
            'features__feature_union__odds_encoder__kw_args': (
                {"book_maker": book_maker, "type_of_bet": type_of_bet},
            ),
            'features__feature_union__game_count_encoder__kw_args': (
                {"last_n": 5}, {"last_n": 6}, {"last_n": 7}, {"last_n": 8},
                {"last_n": 9}, {"last_n": 10}
            ),
            'clf': (
                LogisticRegression(),
                DecisionTreeClassifier(random_state=42),
                ExtraTreeClassifier(random_state=42),
                KNeighborsClassifier(),
                MLPClassifier(random_state=42),
                RandomForestClassifier(random_state=42),
                GradientBoostingClassifier(random_state=42),
                ExtraTreesClassifier(random_state=42),
                AdaBoostClassifier(random_state=42),
                BaggingClassifier(random_state=42),
                RandomForestClassifier(random_state=42),
            )
        }
    if book_maker == "oversea":
        param.pop("features__feature_union__odds_encoder__kw_args")
        param["features__feature_union__odds_encoder"] = ('drop', )
    return param
