# 기본 패키지 import
import warnings
import random
import numpy as np
from time import time
from utils import load_data


warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)


from models.UserKNN_explicit import UserKNN
from models.ItemKNN_explicit import ItemKNN

"""
dataset loading
"""
dataset = "movielens_100k.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=False)

print('train_data size',train_data)
print('valid_data size',valid_data.shape)
print('test_data size',test_data.shape)

# for i, row in enumerate(train_data):
#     print('i',i)
#     print('row',row)
#     train_data[i,np.where(row<0.5)] = np.nan
#     print('glglrow',row)


time_start = time()
userknn_RMSE1 = UserKNN(train=np.copy(train_data), valid=valid_data, test= test_data, top_k=1)
userknn_RMSE5 = UserKNN(train=np.copy(train_data), valid=valid_data, test= test_data, top_k=5)
userknn_RMSE10 = UserKNN(train=np.copy(train_data), valid=valid_data, test= test_data, top_k=10)
userknn_RMSE100 = UserKNN(train=np.copy(train_data), valid=valid_data, test= test_data, top_k=100)
userknn_RMSE500 = UserKNN(train=np.copy(train_data), valid=valid_data, test= test_data, top_k=500)
itemknn_RMSE1 = ItemKNN(train=np.copy(train_data), valid=valid_data, test=test_data, top_k=1)    
itemknn_RMSE5 = ItemKNN(train=np.copy(train_data), valid=valid_data, test=test_data, top_k=5) 
itemknn_RMSE10 = ItemKNN(train=np.copy(train_data), valid=valid_data, test=test_data, top_k=10) 
itemknn_RMSE100 = ItemKNN(train=np.copy(train_data), valid=valid_data, test=test_data, top_k=100) 
itemknn_RMSE500 = ItemKNN(train=np.copy(train_data), valid=valid_data, test=test_data, top_k=500) 

############ PLOT #################################
import matplotlib.pyplot as plt
print("RMSE on Test Data")
print("UserKNN,K=1: %f"%(userknn_RMSE1))
print("UserKNN,K=5: %f"%(userknn_RMSE5))
print("UserKNN,K=10: %f"%(userknn_RMSE10))
print("UserKNN,K=100: %f"%(userknn_RMSE100))
print("UserKNN,K=500: %f"%(userknn_RMSE500))
print("ItemKNN,K=1: %f"%(itemknn_RMSE1))
print("ItemKNN,K=5: %f"%(itemknn_RMSE5))
print("ItemKNN,K=10: %f"%(itemknn_RMSE10))
print("ItemKNN,K=100: %f"%(itemknn_RMSE100))
print("ItemKNN,K=500: %f"%(itemknn_RMSE500))
k = [1,5,10,100,500]
user_rmse = [userknn_RMSE1,userknn_RMSE5,userknn_RMSE10,userknn_RMSE100,userknn_RMSE500]
item_rmse = [itemknn_RMSE1,itemknn_RMSE5,itemknn_RMSE10,itemknn_RMSE100,itemknn_RMSE500]


plt.plot(k,user_rmse)
"""
You should get results as:

RMSE on Test Data
UserKNN: 1.040785
ItemKNN: 1.052665
"""
