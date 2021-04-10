from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

function_transformer = FunctionTransformer(lambda x: x[["away_team", "home_team"]])

team_encoder = Pipeline(steps=[
    ("FT", function_transformer),
    ("OHE", one_hot_encoder),
])


if __name__ =="__main__":
    from data import Data
    
    data = Data(alliance="中國職籃")
    df1 = data.get_train(book_maker="tw", type_of_bet="diff")
    print(team_encoder.fit_transform(df1).toarray())
    print(team_encoder.steps[1][1].categories_)


