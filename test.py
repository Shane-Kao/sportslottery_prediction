if __name__ == '__main__':

    import os

    import dill
    import pandas as pd

    from data import Data
    from configs import MODEL_DIR

    os.system("D:\project\sportslottery_prediction\jobs\crawler.bat")
    alliance = "NBA"
    book_maker = "oversea"
    type_of_bet = "diff"

    target_col = "{}_{}".format(book_maker, type_of_bet)

    model_name = "{}_{}_{}".format(alliance, book_maker, type_of_bet)

    result_dict_ = dill.load(open(os.path.join(MODEL_DIR, model_name), "rb"))
    print(result_dict_['train_size'])
    print(result_dict_['test_size'])
    print(result_dict_['start_date'])
    print(result_dict_['create_time'])
    print(result_dict_['best_score'], result_dict_['p_micro'])

    print('p0', result_dict_['p0'])
    print('p1', result_dict_['p1'])
    print(pd.DataFrame(result_dict_['test_results']))
    model = result_dict_['model']

    data = Data(alliance=alliance)
    df = data.incoming
    df = df[~df[target_col].isnull()]

    df["pred"] = model.predict(df)
    print(df[["game_time", "away_team", "home_team", target_col, "pred", ]])

if __name__ == '__main__xx':
    import os

    import dill
    import pandas as pd

    from configs import DATA_DIR, MODEL_DIR

    models = os.listdir(MODEL_DIR)


    for model in models:
                print(model)
                result_dict_ = dill.load(open(os.path.join(MODEL_DIR, model), "rb"))
                print(result_dict_['train_size'])
                print(result_dict_['test_size'])
                print(result_dict_['start_date'])
                print(result_dict_['create_time'])
                print(result_dict_['best_score'], result_dict_['p_micro'])

                print('p0', result_dict_['p0'])
                print('p1', result_dict_['p1'])
                model = result_dict_["model"]
                print(model.steps[-1][-1])
                # print(pd.DataFrame(result_dict_['test_results']))
                print("=================================================")

if __name__ == '__main__xx':
    from train_jobs import main
    main()
