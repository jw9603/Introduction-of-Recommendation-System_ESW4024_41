from models.SRGNN import SRGNN_session
from models.NISER import NISER_session
from models.SASRec import SASRec_session
from models.CORE import CORE_session
from utils import load_data_session, eval_session, save_submission
from itertools import product
import os
import warnings
import random
import warnings
import torch
import numpy as np
import time
import shutil
warnings.filterwarnings('ignore')


def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


seed = 1
seed_everything(seed)

"""
dataset loading
"""
# music / kaggle
data_name = 'kaggle'
train_df, test_df = load_data_session(data_name=data_name)

"""
model training
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
os.makedirs('saves', exist_ok=True)

# SASRec
# seed_everything(seed)
# start = time.time()
# SASRec = SASRec_session()
# SASRec.fit(train_df, test_df)
# if data_name == 'music':
#     SASRec_R20, SASRec_MRR20 = eval_session(SASRec, train_df, test_df)
#     print(f"[SASRec]\t Test_Recall@20 = {SASRec_R20:.4f} Test_MRR@20 = {SASRec_MRR20:.4f}")
# else:
#     save_submission(SASRec, train_df, test_df)
# print("time sec:", (time.time() - start))
# print("time min:", (time.time() - start)/60.0)
# print("======================================")

# CORE
# seed_everything(seed)
# start = time.time()
# CORE = CORE_session()
# CORE.fit(train_df, test_df)
# if data_name == 'music':
#     CORE_R20, CORE_MRR20 = eval_session(CORE, train_df, test_df)
#     print(f"[CORE]\t Test_Recall@20 = {CORE_R20:.4f} Test_MRR@20 = {CORE_MRR20:.4f}")
# else:
#     save_submission(CORE, train_df, test_df)
# print("time sec:", (time.time() - start))
# print("time min:", (time.time() - start)/60.0)
# print("======================================")

#SRGNN
seed_everything(seed)
start = time.time()
SRGNN = SRGNN_session()
SRGNN.fit(train_df, test_df)
if data_name == 'music':
    SRGNN_R20, SRGNN_MRR20 = eval_session(SRGNN, train_df, test_df)
    print(f"[SRGNN]\t Test_Recall@20 = {SRGNN_R20:.4f} Test_MRR@20 = {SRGNN_MRR20:.4f}")
else:
    save_submission(SRGNN, train_df, test_df)
print("time sec:", (time.time() - start))
print("time min:", (time.time() - start)/60.0)
print("======================================")

## NISER
# define the hyperparameters to search over
# learning_rates = [0.001, 0.01, 0.1]
# hidden_sizes = [32, 64, 128,256]
# l2 = [1e-5,1e-4,1e-3,1e-2]



# #perform grid search
# best_score = 0
# best_params = None
# for lr, hidden_size,l2 in product(learning_rates, hidden_sizes,l2):
#     # create a new instance of the NISER model with the current hyperparameters
#     model = SRGNN_session(lr=lr, hidden_size=hidden_size,l2=l2)

#     # train the model and evaluate on the test set
#     model.fit(train_df, test_df)
#     recall, mrr = eval_session(model, train_df, test_df)

#     # update the best score and best hyperparameters if necessary
#     if recall > best_score:
#         best_score = recall
#         best_params = (lr, hidden_size,l2)

# print(f"Best hyperparameters: learning_rate={best_params[0]}, hidden_size={best_params[1]}, l2={best_params[2]}")
# print(f"Best recall@20: {best_score:.4f}")


# seed_everything(seed)
# start = time.time()
# NISER = NISER_session()
# NISER.fit(train_df, test_df)
# if data_name == 'music':
#     NISER_R20, NISER_MRR20 = eval_session(NISER, train_df, test_df)
#     print(f"[NISER]\t Test_Recall@20 = {NISER_R20:.4f} Test_MRR@20 = {NISER_MRR20:.4f}")
# else:
#     save_submission(NISER, train_df, test_df)
# print("time sec:", (time.time() - start))
# print("time min:", (time.time() - start)/60.0)
# print("======================================")

if data_name == 'kaggle':
    # print(f"[SASRec]\t Test_Recall@20 = {SASRec_R20:.4f} Test_MRR@20 = {SASRec_MRR20:.4f}")
    # print(f"[CORE]\t Test_Recall@20 = {CORE_R20:.4f} Test_MRR@20 = {CORE_MRR20:.4f}")
    print(f"[SRGNN]\t Test_Recall@20 = {SRGNN_R20:.4f} Test_MRR@20 = {SRGNN_MRR20:.4f}")
    # print(f"[NISER]\t Test_Recall@20 = {NISER_R20:.4f} Test_MRR@20 = {NISER_MRR20:.4f}")
shutil.rmtree('saves')