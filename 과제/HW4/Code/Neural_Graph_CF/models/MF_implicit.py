import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from utils import eval_implicit

class MF_implicit_model(torch.nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users, n_features, sparse=False)
        self.item_factors = torch.nn.Embedding(num_items, n_features, sparse=False)
        self.user_bias = torch.nn.Embedding(num_users, 1, sparse=False)
        self.item_bias = torch.nn.Embedding(num_items, 1, sparse=False)

        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)
        torch.nn.init.normal_(self.user_bias.weight, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self):
        reconstruction = None
        reconstruction = torch.matmul(self.user_factors.weight, self.item_factors.weight.T)
        reconstruction = reconstruction + self.user_bias.weight # add user bias
        reconstruction = reconstruction + self.item_bias.weight.T # add item bias
        return reconstruction


class MF_implicit():
    def __init__(self, train, valid, n_features=20, learning_rate = 1e-2, reg_lambda =0.1, num_epochs = 100, device='cpu'):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features
        self.device = device

        self.model = MF_implicit_model(self.num_users, self.num_items, self.n_features).to(device)
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    def fit(self):
        ratings = torch.FloatTensor(self.train).to(self.device)

        # U와 V를 업데이트 함.
        for epoch in range(self.num_epcohs):
            self.optimizer.zero_grad()

            # 예측
            prediction = self.model.forward()
            loss = self.BCE_loss(prediction, ratings)

            # Backpropagate
            loss.backward()

            # Update the parameters
            self.optimizer.step()

            if epoch % 20 == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.reconstructed = self.model.forward().cpu().numpy()
                    self.model.train()

                top_k=50
                print("[MF] epoch %d, loss: %f"%(epoch, loss))
                prec, recall, ndcg = eval_implicit(self, self.train, self.valid, top_k)
                print(f"(MF VALID) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}")


        with torch.no_grad():
            self.model.eval()
            self.reconstructed = self.model.forward().cpu().numpy()
        
    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]