from sklearn.preprocessing import FunctionTransformer


def func(df, type_of_bet):
    df_ = df[
        ["home_win_pct", "away_win_pct", "home_win_pct_is_higher"] +
        ["away_avg_{}".format(type_of_bet), "home_avg_{}".format(type_of_bet)]
    ]
    return df_.values.tolist()


stats_encoder = FunctionTransformer(
    func=func,
    kw_args={"type_of_bet": "diff"}
)


if __name__ == "__main__":
    from data import Data

    data = Data(alliance="NBA")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    # print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
    #                      "is_back_to_back", ]])
    print(stats_encoder.fit_transform(df1))
    # print(game_count_encoder.steps[1][1].categories_)

