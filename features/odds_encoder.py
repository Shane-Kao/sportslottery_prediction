from sklearn.preprocessing import FunctionTransformer


def func(df, book_maker, type_of_bet):
    if book_maker == "tw":
        if type_of_bet == "diff":
            cols = ['tw_diff_away_odds', 'tw_diff_home_odds']
            df_ = df[cols]
            result_list = [[i[0], i[1], 1 if i[0] < i[1] else 0] for i in df_.values.tolist()]
        elif type_of_bet == "total":
            cols = ['tw_under_odds', 'tw_over_odds']
            df_ = df[cols]
            result_list = [[i[0], i[1], 1 if i[0] < i[1] else 0] for i in df_.values.tolist()]
        else:
            raise ValueError
    else:
        result_list = [[0, 0, 0] for _ in range(df.shape[0])]
    return result_list


odds_encoder = FunctionTransformer(
    func=func,
    kw_args={"book_maker": "oversea", "type_of_bet": "diff"}
)


if __name__ == "__main__":
    from data import Data

    data = Data(alliance="日本職棒")
    df1 = data.get_train(book_maker="oversea", type_of_bet="diff")
    print(df1)
    # print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
    #                      "is_back_to_back", ]])
    print(odds_encoder.fit_transform(df1))
    # print(game_count_encoder.steps[1][1].categories_)

