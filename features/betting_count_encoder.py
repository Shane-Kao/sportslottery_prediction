from sklearn.preprocessing import FunctionTransformer


def func(df, book_maker, type_of_bet):
    col = "{}_over_count".format(book_maker) if type_of_bet == "total" else "{}_diff_home_count".format(book_maker)
    cols = [col]
    df_ = df[cols]
    return df_.values.tolist()


betting_count_encoder = FunctionTransformer(
    func=func,
    kw_args={"book_maker": "oversea", "type_of_bet": "total"}
)


if __name__ == "__main__":
    from data import Data

    data = Data(alliance="NBA")
    df1 = data.get_train(book_maker="oversea", type_of_bet="total")
    # print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
    #                      "is_back_to_back", ]])
    print(betting_count_encoder.fit_transform(df1))
    # print(game_count_encoder.steps[1][1].categories_)

