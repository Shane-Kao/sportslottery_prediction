from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

one_hot_encoder = OrdinalEncoder()

function_transformer = FunctionTransformer(lambda x: x[["away_team_is_back_to_back", "home_team_is_back_to_back"]])

back_to_back_encoder = Pipeline(steps=[
    ("FT", function_transformer),
    ("OHE", one_hot_encoder),
])

if __name__ == "__main__":
    from data import Data

    data = Data(alliance="SBL")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    print(back_to_back_encoder.fit_transform(df1))
    print(back_to_back_encoder.steps[1][1].categories_)

