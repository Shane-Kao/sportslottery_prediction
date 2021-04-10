from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

function_transformer = FunctionTransformer(
    lambda x: x[["is_back_to_back",]]
)

back_to_back_encoder = Pipeline(steps=[
    ("FT", function_transformer),
    ("OHE", one_hot_encoder),
])

if __name__ == "__main__":
    from data import Data

    data = Data(alliance="中國職籃")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    print(df1[["game_time", "away_team", "home_team", "away_score", "home_score",
                         "is_back_to_back", ]])
    print(back_to_back_encoder.fit_transform(df1).toarray())
    print(back_to_back_encoder.steps[1][1].categories_)

