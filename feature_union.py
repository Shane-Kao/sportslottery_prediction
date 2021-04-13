from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from features.back_to_back_encoder import back_to_back_encoder
from features.team_encoder import team_encoder
from features.weekday_encoder import weekday_encoder

feature_union = FeatureUnion([
    ("weekday_encoder", weekday_encoder),
    ("team_encoder", team_encoder),
    ("back_to_back_encoder", back_to_back_encoder),
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

    data = Data(alliance="NBA")
    df1 = data.get_train(book_maker="tw", type_of_bet="total")

    X = features.fit_transform(X=df1)
    print(X.shape)
