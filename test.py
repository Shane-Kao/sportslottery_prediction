import os

import dill
import pandas as pd

from data import Data
from configs import MODEL_DIR

alliance = "歐洲職籃"
book_maker = "tw"
type_of_bet = "diff"

target_col = "{}_{}".format(book_maker, type_of_bet)

model_name = "{}_{}_{}".format(alliance, book_maker, type_of_bet)

result_dict_ = dill.load(open(os.path.join(MODEL_DIR, model_name), "rb"))
print(result_dict_['start_date'])
print(result_dict_['create_time'])
print(result_dict_['best_score'], result_dict_['p_micro'])

print(result_dict_['p0'])
print(result_dict_['p1'])
print(pd.DataFrame(result_dict_['test_results']))
model = result_dict_['model']

data = Data(alliance=alliance)
df = data.incoming
df = df[~df[target_col].isnull()]

df["pred"] = model.predict(df)
print(df[["game_time", "away_team", "home_team", target_col, "pred", ]])