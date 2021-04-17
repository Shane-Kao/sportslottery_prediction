from sklearn.preprocessing import FunctionTransformer


def _select_last_n(df, last_n):
    df_ = df[
        ["home_team_game_count_last{}".format(last_n),
         "away_team_game_count_last{}".format(last_n)]
    ]
    return df_.values.tolist()


game_count_encoder = FunctionTransformer(
    func=_select_last_n,
    kw_args={"last_n": 5}
)


if __name__ == "__main__":
    from data import Data

    data = Data(alliance="中華職棒")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    # print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
    #                      "is_back_to_back", ]])
    print(game_count_encoder.fit_transform(df1))
    # print(game_count_encoder.steps[1][1].categories_)

