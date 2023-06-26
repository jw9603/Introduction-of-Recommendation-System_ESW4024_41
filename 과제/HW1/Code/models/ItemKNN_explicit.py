import numpy as np
from tqdm import tqdm
from time import time
from sklearn.metrics import mean_squared_error

def ItemKNN(train, valid, test, top_k):
    num_users = train.shape[0]
    num_items = train.shape[1]

    for i, row in enumerate(train):
        train[i, np.where(row < 0.5)[0]] = np.nan

    user_mean = np.nanmean(train, axis=1)
    user_mean[np.isnan(user_mean)] = 0.0
    train = train - user_mean[:, None]

    item_item_sim_matrix = np.zeros((num_items, num_items))

    """
    model training
    """

    print("model training...")
    time_start = time()

    ############## EDIT HERE ####################################
    for item_i in tqdm(range(0,num_items),desc='item_item_sim_matrix (k=%d)'% top_k):
        for item_j in range(item_i+1,num_items):
            a = train.T[item_i]
            b = train.T[item_j]
            
            co_rated = ~np.logical_or(np.isnan(a),np.isnan(b))
            a = np.compress(co_rated,a)
            b = np.compress(co_rated,b)
            
            if len(a)==0:
                continue
            
            
            dot_a_b = np.dot(a,b)
            if dot_a_b == 0:
                continue
            item_item_sim_matrix[item_i, item_j] = dot_a_b / (np.linalg.norm(a) * np.linalg.norm(b))
    item_item_sim_matrix = (item_item_sim_matrix + item_item_sim_matrix.T)
    #############################################################
    
    
    print("training time: ", time()-time_start)

    """
    model evaluation
    """
    print("model evaluation")
    train_data = train+valid
    test_data = test
    time_start = time()
    ori_top_k = top_k
    rmse_list = []

    num_users, num_items = train_data.shape
    pred_matrix = np.zeros((num_users, num_items))

    for item_id in range(len(train_data.T)):
        train_by_item = test_data[:, item_id] # 한 row씩 진행
        missing_user_ids = np.where(train_by_item >= 0.5)[0]
        
        predicted_values = []

        for one_missing_user in missing_user_ids:
            ################# EDIT HERE ################################
            # user i가 시청한 item들
            rated_items = np.where(~np.isnan(train.T[:,one_missing_user]))[0]
            unsorted_sim = item_item_sim_matrix[item_id,rated_items]
            
            # 유사도 정렬
            sorted_items = np.argsort(unsorted_sim)
            sorted_items = sorted_items[::-1]

            # Top K 이웃 구하기
            if ori_top_k > len(sorted_items):
                top_k = len(sorted_items)
            else:
                top_k = ori_top_k
            sorted_items = sorted_items[0:top_k]
            top_k_items = rated_items[sorted_items]
            # 예측 값 구하기
            if top_k == 0:
                predicted_values.append(0.0)
            else:
                items_rate = train[one_missing_user,top_k_items] + user_mean[one_missing_user]
                items_sim = item_item_sim_matrix[item_id,top_k_items]
                items_sim[items_sim < 0.0] = 0.0
                # print('items_rate',items_rate)
                # print('items_rate',items_rate.shape)
              
                
                if np.sum(items_sim) == 0.0:
                    predicted_rate = np.sum(items_rate)/len(items_rate)
                else:
                    predicted_rate =  np.sum(items_sim*items_rate)/np.sum(items_sim)
                
                
                # if np.isnan(predicted_rate):
                #     predicted_rate = 0
                if predicted_rate < 1:
                    predicted_rate = 1
                elif predicted_rate > 5:
                    predicted_rate = 5
                predicted_values.append(predicted_rate)
            # predicted_values = None
            #############################################################
            

        pred_matrix[missing_user_ids, item_id] = predicted_values

    for user_id in range(len(train_data)):
        test_by_user = test_data[user_id]
        target_u = np.where(test_by_user >= 0.5)[0]
        target_u_score = test_by_user[target_u]

        pred_u_score = pred_matrix[user_id, target_u]

        rmse = mean_squared_error(target_u_score, pred_u_score)
        rmse_list.append(rmse)

    return np.mean(rmse_list)


