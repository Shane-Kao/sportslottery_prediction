from sklearn.preprocessing import FunctionTransformer


def func(df, book_maker, type_of_bet):
    cols = [
        "away_strk",
        "home_strk",
        "away_strk_{}_diff".format(book_maker),
        "home_strk_{}_diff".format(book_maker),
    ] if type_of_bet == "diff" else [
        "away_strk_{}_total".format(book_maker),
        "home_strk_{}_total".format(book_maker),
    ]
    df_ = df[cols]
    return df_.values.tolist()


strk_encoder = FunctionTransformer(
    func=func,
    kw_args={"book_maker": "tw", "type_of_bet": "total"}
)


if __name__ == "__main__":
    from data import Data

    data = Data(alliance="中華職棒")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    # print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
    #                      "is_back_to_back", ]])
    print(strk_encoder.fit_transform(df1))
    # print(game_count_encoder.steps[1][1].categories_)

