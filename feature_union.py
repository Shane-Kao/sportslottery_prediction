from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from features.back_to_back_encoder import back_to_back_encoder
from features.team_encoder import team_encoder
from features.strk_encoder import strk_encoder
from features.game_count_encoder import game_count_encoder
from features.weekday_encoder import weekday_encoder
from features.stats_encoder import stats_encoder

feature_union = FeatureUnion(
    transformer_list=[
    ("weekday_encoder", weekday_encoder),
    ("team_encoder", team_encoder),
    ("back_to_back_encoder", back_to_back_encoder),
    ("game_count_encoder", game_count_encoder),
    ("strk_encoder", strk_encoder),
    ("stats_encoder", stats_encoder)
], transformer_weights=None)

polynomial_features = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=True
)

variance_threshold = VarianceThreshold(threshold=0)

features = Pipeline(steps=[
    ("feature_union", feature_union),
    ("polynomial_features", polynomial_features),
    ("variance_threshold", variance_threshold),
])


if __name__ == "__main__":

    from data import Data

    data = Data(alliance="中華職棒")
    df1 = data.get_train(book_maker="tw", type_of_bet="total")

    X = feature_union.fit_transform(X=df1)
    print(X.shape)
