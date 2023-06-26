import pandas as pd

from scipy import sparse
from sklearn.model_selection import train_test_split


def load_data(data_name, implicit=True):
    data_path = './data/%s' % (data_name)

    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    movie_data = pd.read_csv(data_path, names=column_names)

    if implicit:
        movie_data['rating'] = 1

    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    user_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(user_list)}
    movie_data.user_id = [user_id_dict[x] for x in movie_data.user_id.tolist()]

    item_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(item_list)}
    movie_data.item_id = [item_id_dict[x] for x in movie_data.item_id.tolist()]
    print(f"# of users: {num_users},  # of items: {num_items},  # of ratings: {num_ratings}")

    movie_data = movie_data[['user_id', 'item_id', 'rating']]
    movie_data = movie_data.sort_values(by="user_id", ascending=True)
    
    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify=movie_data['user_id'], random_state=1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify=train_valid['user_id'], random_state=1234)

    train = train.to_numpy()
    valid = valid.to_numpy()
    test = test.to_numpy()

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in train:
        matrix[u, i] = r
    train = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in valid:
        matrix[u, i] = r
    valid = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in test:
        matrix[u, i] = r
    test = sparse.csr_matrix(matrix)

    return train.toarray(), valid.toarray(), test.toarray()

