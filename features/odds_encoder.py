from sklearn.preprocessing import FunctionTransformer


def func(df, book_maker, type_of_bet):
    if book_maker == "tw":
        if type_of_bet == "diff":
            cols = ['tw_diff_away_odds', 'tw_diff_home_odds']
            df_ = df[cols]
            df_["home_odds_is higher"] = df_['tw_diff_home_odds'] > df_['tw_diff_away_odds']
        elif type_of_bet == "total":
            cols = ['tw_under_odds', 'tw_over_odds']
            df_ = df[cols]
            df_["over_odds_is higher"] = df_['tw_over_odds'] > df_['tw_under_odds']
        else:
            raise ValueError
    else:
        return None
    return df_.values.tolist()


odds_encoder = FunctionTransformer(
    func=func,
    kw_args={"book_maker": "tw", "type_of_bet": "total"}
)


if __name__ == "__main__":
    from data import Data

    data = Data(alliance="中華職棒")
    df1 = data.get_train(book_maker="tw", type_of_bet="total")
    # print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
    #                      "is_back_to_back", ]])
    print(odds_encoder.fit_transform(df1))
    # print(game_count_encoder.steps[1][1].categories_)

