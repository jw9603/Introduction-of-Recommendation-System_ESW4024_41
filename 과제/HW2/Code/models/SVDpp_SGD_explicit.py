import numpy as np
from tqdm import tqdm

class SVDpp_explicit():
    def __init__(self, train, valid, n_features=20, learning_rate = 1e-2, reg_lambda =0.1, num_epochs = 100):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.n_features = n_features

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        self.y = np.zeros_like(self.train)
        for i, row in enumerate(self.train):
            self.y[i, np.where(row > 0.5)[0]] = 1.0
            

        # ========================= EDIT HERE ========================
        # Get the non-zero index of user, item from the ratings matrix
        self.user_indices, self.item_indices = np.nonzero(self.train)
        self.num_ratings = len(self.user_indices)


        # Add bias terms
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)
        self.bias = np.mean(self.train[np.where(self.train!=0)])
        
        self.user_factors = np.random.normal(scale=1/self.n_features,size=(self.num_users,self.n_features))
        self.item_factors = np.random.normal(scale=1/self.n_features,size=(self.num_items,self.n_features))
        self.item_factors_y = np.random.normal(scale=1/self.n_features,size=(self.num_items,self.n_features))


        # ========================= EDIT HERE ========================


    def mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()


    def fit(self):
        ratings = self.train
        weights = self.y
        
        print(f"> Training SVD++ with SGD for {self.num_epochs} epochs")
        # ========================= EDIT HERE ========================
        # Get the implicit ratings
        self.item_weights = np.zeros(self.num_items)
        for i in range(self.num_items):
            self.item_weights[i] = np.sqrt(np.sum(self.y[:,i]))
        
        
        # ========================= EDIT HERE ========================

        for epoch in tqdm(range(self.num_epochs), dynamic_ncols=True):
            # Shuffle the data
            indices = np.random.permutation(self.num_ratings)

            # For each observed entries
            for idx in indices:
                
                # Get the user and item index
                user_id = self.user_indices[idx]
                item_id = self.item_indices[idx]
                # ========================= EDIT HERE ========================
                # Compute the errors (Use the predict_single_entry function)
                error = self.train[user_id,item_id] - self.predict_single_entry(user_id, item_id)

                
                # Update biases
                self.user_bias[user_id] += self.learning_rate * (error-self.reg_lambda * self.user_bias[user_id])
                self.item_bias[item_id] += self.learning_rate * (error-self.reg_lambda * self.item_bias[item_id])
                
                
                # Update the factors
                tmp = self.item_factors[item_id,:].copy()
                self.item_factors[item_id,:] += self.learning_rate * (error * self.user_factors[user_id,:] - self.reg_lambda * self.item_factors[item_id,:])
                self.user_factors[user_id,:] += self.learning_rate * (error * (tmp + self.item_factors_y[item_id,:]) / self.item_weights[item_id] - self.reg_lambda * self.user_factors[user_id,:])
                self.item_factors_y[item_id,:] += self.learning_rate *(error * tmp / self.item_weights[item_id] - self.reg_lambda * self.item_factors_y[item_id,:])

                # ========================= EDIT HERE ========================
                
            # ========================= EDIT HERE ========================
            # Compute the loss (Use the predict_matrix function)

            loss = self.mse_loss(weights, ratings, self.predict_matrix())

            # ========================= EDIT HERE ========================

            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss: {loss}")

        self.reconstructed = self.predict_matrix()


    def predict_single_entry(self, user_id, item_id):
        prediction = None
        # ========================= EDIT HERE ========================
        prediction = self.bias + self.user_bias[user_id] + self.item_bias[item_id] + np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :] + self.item_factors_y[item_id, :] / self.item_weights[item_id])

        # ========================= EDIT HERE ========================
        return prediction


    def predict_matrix(self):
        reconstructed = None
        # ========================= EDIT HERE ========================
        # print(self.bias.shape)
        # reconstructed = self.bias + self.user_bias[:, np.newaxis] + self.item_bias[np.newaxis:,] + np.dot(self.user_factors, (self.item_factors.T + self.item_factors_y.T / self.item_weights).T)
        reconstructed = self.bias + self.user_bias[:,np.newaxis] + self.item_bias[np.newaxis:,] + np.dot(self.user_factors,self.item_factors.T)

        # ========================= EDIT HERE ========================
        return reconstructed


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
