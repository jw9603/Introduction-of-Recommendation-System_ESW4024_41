import warnings
import random
import numpy as np

from time import time

from utils import load_data, eval_explicit
from models.MF_SGD_explicit import MF_explicit
from models.BiasedMF_SGD_explicit import BiasedMF_explicit
from models.SVDpp_SGD_explicit import SVDpp_explicit

warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)

"""
dataset loading
"""
dataset = "movielens_100k.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=False)


"""
model training
"""
print('train_data',train_data.shape)
print('valid_data',valid_data.shape)
print(">> Model training ...")
biasedmf = BiasedMF_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)
mf = MF_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)

svdpp = SVDpp_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)

st = time()
biasedmf.fit()
t1 = time()
mf.fit()
t2 = time()
svdpp.fit()
t3 = time()
print(f"Training time: {time() - st:.2f} (MF: {t1 - st:.2f}, BiasedMF: {t2 - t1:.2f}, SVD++: {t3 - t2:.2f})")

"""
model evaluation
"""
st = time()
print(">> Model evaluating ...")
biasedmf_rmse = eval_explicit(biasedmf, train_data, test_data)
mf_rmse = eval_explicit(mf, train_data, test_data)
svdpp_rmse = eval_explicit(svdpp, train_data, test_data)
print("Evaluation time: ", time()-st)

print("RMSE on test data")
print(f"MF: {mf_rmse}")
print(f"BiasedMF: {biasedmf_rmse}")
print(f"SVD++: {svdpp_rmse}")

"""
You should get results as:
RMSE on Test Data
SVD: 1.019556
MF: 0.919983
BiasedMF: 0.900350
SVD++: 0.900527
"""
