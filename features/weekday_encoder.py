from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

function_transformer = FunctionTransformer(lambda x: x["game_time"].dt.dayofweek.to_frame())

weekday_encoder = Pipeline(steps=[
    ("FT", function_transformer),
    ("OHE", one_hot_encoder),
])

if __name__ == "__main__":
    from data import Data

    data = Data(alliance="SBL")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    print(weekday_encoder.fit_transform(df1).toarray())
    print(weekday_encoder.steps[1][1].categories_)

