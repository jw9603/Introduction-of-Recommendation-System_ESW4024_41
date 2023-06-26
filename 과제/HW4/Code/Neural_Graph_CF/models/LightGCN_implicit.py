
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import numpy as np
import scipy.sparse as sp
from IPython import embed
from utils import eval_implicit


class LightGCN_implicit(nn.Module):
    def __init__(self, train, valid, learning_rate, regs, batch_size, num_epochs, emb_size, num_layers, node_dropout, device='cpu'):
        super(LightGCN_implicit, self).__init__()
        
        self.train_data = train
        self.valid_data = valid

        self.train_mat = sp.csr_matrix(train)
        self.valid_mat = sp.csr_matrix(valid)

        self.num_users, self.num_items = self.train_mat.shape

        self.R = sp.csr_matrix(train)

        self.norm_adj = self.create_adj_mat()

        self.learning_rate = learning_rate
        self.device = device
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.node_dropout = node_dropout

        self.decay = regs
        
        self.embedding_dict = self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items, self.emb_size)))
        })
        
        return embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

        return out * (1. / (1 - rate))

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=False):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]


        '''
        Implement GCN process
        '''
        # ========================= EDIT HERE ========================  
        # Light Graph Convolutional Operations
        for k in range(self.num_layers):
            # Message Aggregation
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            sum_embeddings =F.leaky_relu(side_embeddings)

            # Store k-th embeddings
            ego_embeddings = sum_embeddings
            all_embeddings += [ego_embeddings]
        # ========================= EDIT HERE ========================  

        all_embeddings = torch.stack(all_embeddings, 1)
        final_embeddings = torch.mean(all_embeddings, 1)

        u_g_embeddings = final_embeddings[:self.num_users, :]
        i_g_embeddings = final_embeddings[self.num_users:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, i_g_embeddings

    def fit(self):
        user_idx = np.arange(self.num_users)

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            self.train()

            np.random.RandomState(12345).shuffle(user_idx)

            batch_num = int(len(user_idx) / self.batch_size) + 1

            for batch_idx in range(batch_num):
                batch_users = user_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                batch_matrix = torch.FloatTensor(self.train_mat[batch_users, :].toarray()).to(self.device)
                batch_users = torch.LongTensor(batch_users).to(self.device)
                batch_loss = self.train_model_per_batch(batch_matrix, batch_users)

                if torch.isnan(batch_loss):
                    print('Loss NAN. Train finish.')
                    break
                
                epoch_loss += batch_loss
            

            if epoch % 20 == 0:
                with torch.no_grad():
                    self.eval()
                    
                    top_k=50
                    print("[LightGCN] epoch %d, loss: %f"%(epoch, epoch_loss))

                    prec, recall, ndcg = eval_implicit(self, self.train_data, self.valid_data, top_k)
                    print(f"(LightGCN VALID) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}")
                    self.train()

    def train_model_per_batch(self, train_matrix, batch_users, pos_items=0, neg_items=0):
        self.optimizer.zero_grad()

        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, i_g_embeddings = self.forward(batch_users, 0, 0)


        # Implement output and loss function (binary CE loss)
        # ========================= EDIT HERE ========================
        output = self.rating(u_g_embeddings, pos_i_g_embeddings)
        loss = F.binary_cross_entropy_with_logits(output, torch.ones_like(output))
        loss += F.binary_cross_entropy_with_logits(output, torch.zeros_like(output))    
        
        neg_output = self.rating(u_g_embeddings, neg_i_g_embeddings)
        loss += F.binary_cross_entropy_with_logits(neg_output, torch.zeros_like(neg_output))
        for embeeding in i_g_embeddings:
            loss += self.decay * torch.mean(embeeding.pow(2))    
        # ========================= EDIT HERE ========================  

        loss.backward()
        self.optimizer.step()

        return loss


    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            u_g_embeddings, _, _, i_g_embeddings = self.forward(user_ids, 0, 0)

            # ========================= EDIT HERE ========================
            output = self.rating(u_g_embeddings, i_g_embeddings)
            
            # ========================= EDIT HERE ========================  
    
            predict_ = output.detach().cpu().numpy()
            return predict_[item_ids]


    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)

        adj_mat = adj_mat.tolil()
        R = sp.csr_matrix(self.R).tolil()

        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = norm_adj.tocsr()

        return norm_adj

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))