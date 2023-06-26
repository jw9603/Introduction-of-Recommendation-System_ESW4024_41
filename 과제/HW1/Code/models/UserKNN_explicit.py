import numpy as np
from tqdm import tqdm
from time import time
from sklearn.metrics import mean_squared_error

def UserKNN(train, valid, test, top_k):
    num_users = train.shape[0]
    num_items = train.shape[1]

    for i, row in enumerate(train):
        # i = row index
        # row = row value
        train[i, np.where(row < 0.5)[0]] = np.nan
        # Values less than 0.5 in each row are treated as nan.

    user_mean = np.nanmean(train, axis=1)
    user_mean[np.isnan(user_mean)] = 0.0
    train = train - user_mean[:, None] # zero-centering

    user_user_sim_matrix = np.zeros((num_users, num_users))

    """
    model training
    """

    print("model training...")
    time_start = time()

    for user_i in tqdm(range(0, num_users), desc='user_user_sim_matrix (k=%d)' % top_k):
        for user_j in range(user_i+1, num_users):
            a = train[user_i]
            b = train[user_j]

            co_rated = ~np.logical_or(np.isnan(a), np.isnan(b))
            a = np.compress(co_rated, a)
            b = np.compress(co_rated, b)

            if len(a) == 0:
                continue 

            dot_a_b = np.dot(a, b)
            if dot_a_b == 0:
                continue

            user_user_sim_matrix[user_i, user_j] = dot_a_b / (np.linalg.norm(a) * np.linalg.norm(b))
    # print('user_user_sim_matrix 사이즈',user_user_sim_matrix)
    # assert -1 == 0
    user_user_sim_matrix = (user_user_sim_matrix + user_user_sim_matrix.T)

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
    
    for user_id in range(len(train_data)):
        test_by_user = test_data[user_id]
        target_u = np.where(test_by_user >= 0.5)[0]
        target_u_score = test_by_user[target_u]
        
        predicted_values=[]

        for one_missing_item in target_u:
            # item i를 시청한 사용자들
            rated_users = np.where(~np.isnan(train[:, one_missing_item]))[0]
            unsorted_sim = user_user_sim_matrix[user_id, rated_users]

            # 유사도 정렬
            sorted_users = np.argsort(unsorted_sim)
            sorted_users = sorted_users[::-1]

            # Top K 이웃 구하기
            if ori_top_k > len(sorted_users):
                top_k = len(sorted_users)
            else:
                top_k = ori_top_k
            sorted_users = sorted_users[0:top_k]
            top_k_users = rated_users[sorted_users]

            # 예측 값 구하기
            if top_k == 0:
                predicted_values.append(0.0)
            else:
                users_rate = train[top_k_users, one_missing_item]
                users_sim = user_user_sim_matrix[user_id, top_k_users]
                users_sim[users_sim < 0.0] = 0.0
             
                if np.sum(users_sim) == 0.0:
                    predicted_rate = user_mean[user_id]
                else:
                    predicted_rate = user_mean[user_id] + np.sum(users_rate*users_sim)/np.sum(users_sim)
                # 1미만 5초과 값 처리
                if predicted_rate < 1:
                    predicted_rate = 1
                elif predicted_rate > 5:
                    predicted_rate = 5
                predicted_values.append(predicted_rate)

        rmse = mean_squared_error(target_u_score, predicted_values)
        rmse_list.append(rmse)
    print("evaluation time: ", time()-time_start)
    return np.mean(rmse_list)    


