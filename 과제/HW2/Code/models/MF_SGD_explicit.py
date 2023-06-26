import numpy as np

class MF_explicit():
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
            
        self.user_factors = np.random.randn(self.num_users, self.n_features) * 0.01
        self.item_factors = np.random.randn(self.num_items, self.n_features) * 0.01

        # Get the non-zero index of user, item from the ratings matrix
        self.user_indices = np.nonzero(self.train)[0]
        self.item_indices = np.nonzero(self.train)[1]
        self.num_ratings = len(self.user_indices)


    def mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()


    def fit(self):
        ratings = self.train
        weights = self.y
        print(f"> Training MF with SGD for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):

            # Shuffle the data
            indices = np.random.permutation(self.num_ratings)

            # For each observed entries
            for idx in indices:

                # Get the user and item index
                user_id = self.user_indices[idx]
                item_id = self.item_indices[idx]

                # Compute the errors
                error = ratings[user_id, item_id] - np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :].T)

                # Update the factors
                gradient_u = error * self.item_factors[item_id, :] - self.reg_lambda * self.user_factors[user_id, :]
                gradient_v = error * self.user_factors[user_id, :] - self.reg_lambda * self.item_factors[item_id, :]

                self.user_factors[user_id, :] += self.learning_rate * gradient_u
                self.item_factors[item_id, :] += self.learning_rate * gradient_v
                
            # Compute the loss
            loss = self.mse_loss(weights, ratings, np.dot(self.user_factors, self.item_factors.T))
            
            if epoch % 10 == 0:
                print("epoch %d, loss: %f"%(epoch, loss))

        self.reconstructed = np.dot(self.user_factors, self.item_factors.T)


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
