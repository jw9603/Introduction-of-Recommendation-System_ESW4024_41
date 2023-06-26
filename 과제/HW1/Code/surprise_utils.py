import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

# https://data-science-hi.tistory.com/80
model_farm = {
    "KNNBasic" :KNNBasic,
    "KNNBaseline" :KNNBaseline,
    "KNNWithMeans":KNNWithMeans,
    "KNNWithZScore":KNNWithZScore,
    "SVD":SVD,
    "SVD++":SVDpp,
    "NMF":NMF,
    "SlopeOne":SlopeOne,
    "CoClustering":CoClustering
} # https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html

def load_train_data(file, valid_ratio):
    train, valid = None, None

    reader = Reader(
        line_format="user item rating",
        rating_scale=(0.0, 10.0),
        sep=',',
        skip_lines=1
    )
    data = Dataset.load_from_file(file, reader=reader)
    
    if valid_ratio > 0.0:
        train, valid = train_test_split(data, test_size=valid_ratio)
    else:
        train = data.build_full_trainset()

    return train, valid

def load_test_data(file):
    data = pd.read_csv(file)
    empty_rating = np.zeros(data.shape[0])
    test = list(zip(data['user_id'], data['item_id'], empty_rating))

    return test

def fit(train, valid, model, config):

    model = model_farm[model](**config)
    model.fit(train)

    print("Train complete")

    if valid is not None:
        pred = model.test(valid)
        print(f"Valid RMSE: {accuracy.rmse(pred)}")
        print(f"Valid MSE: {accuracy.mse(pred)}")
        print(f"Valid MAE: {accuracy.mae(pred)}")
        print(f"Valid FCP: {accuracy.fcp(pred)}")
    return model
    
def predict(model, test, file=None):
    num_test = len(test)
    result = model.test(test)
    pred = np.array([result[i].est for i in range(len(result))])

    if file is not None:
        data_to_save = pd.DataFrame(zip(range(1, num_test+1), pred), columns=['test_id', 'rating'])
        data_to_save.to_csv(file, index=False)
        print("Submission clear")

    return pred