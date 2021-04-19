import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import make_scorer, classification_report, accuracy_score, roc_auc_score, precision_score, confusion_matrix

from feature_union import features
from data import Data


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    book_maker = "tw"
    type_of_bet = "diff"
    data = Data(alliance="日本職棒")
    df = data.get_train(book_maker=book_maker, type_of_bet=type_of_bet)
    y = df["{}_{}_result".format(book_maker, type_of_bet)].astype(int)

    test_size = int(df.shape[0] * 0.1)
    test_size = 10 if test_size < 10 else 100 if test_size > 100 else test_size
    train_size = df.shape[0] - test_size
    print(train_size, test_size)

    pipeline = Pipeline([
        ('features', features),
        ('feature_selection', SelectPercentile()),
        ('clf', LogisticRegression())
    ])

    test_fold = [-1 for _ in range(train_size)] + [0 for _ in range(test_size)]
    ps = PredefinedSplit(test_fold)
    param = {
        'clf__fit_intercept': (True, False),
        'feature_selection__percentile': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                                          65, 70, 75, 80, 85, 90, 95, 100],
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
            {"last_n": 5}, {"last_n": 6}, {"last_n": 7}, {"last_n": 8}, {"last_n": 9}, {"last_n": 10}
        )
    }

    if book_maker == "oversea":
        param.pop("features__feature_union__odds_encoder__kw_args")
        param["features__feature_union__odds_encoder"] = ('drop', )

    # scoring = {
    #
    #     'AUC': make_scorer(roc_auc_score),
    #     'precision_pos': make_scorer(precision_score, pos_label=1),
    #     'precision_neg': make_scorer(precision_score, pos_label=0)
    # }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param,
        n_jobs=-1,
        verbose=1,
        cv=ps,
        scoring=make_scorer(precision_score, average="weighted"),
        refit=False,
    )

    grid_search.fit(df, y)

    print("Best precision score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:", grid_search.best_params_)


    for train_index, test_index in ps.split():
        X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index, ], y.iloc[test_index, ]

    pipeline.set_params(**grid_search.best_params_)

    pipeline.fit(X=X_train, y=y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    # print(roc_auc_score(y_true=y_test, y_score=y_pred_proba[:, 0]))
    print(precision_score(y_true=y_test, y_pred=y_pred, average="weighted"))

    y_pred_proba = pipeline.predict_proba(X_test)
    # print(roc_auc_score(y_true=y_test, y_score=y_pred))
    # print("acc", accuracy_score(y_true=y_test, y_pred=y_pred))
    print("p0", precision_score(y_true=y_test, y_pred=y_pred, pos_label=0))
    print("p1", precision_score(y_true=y_test, y_pred=y_pred, pos_label=1))

    print(sorted(zip(y_test, y_pred_proba[:, 1]), key=lambda x: x[1]))
