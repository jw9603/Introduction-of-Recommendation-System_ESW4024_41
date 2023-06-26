import os
import math
import datetime
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F


class NISER(nn.Module):
    def __init__(self, hidden_size, batch_size, nonhybrid, step, lr, l2, lr_dc, lr_dc_step, n_node, max_pos):
        super(NISER, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nonhybrid = nonhybrid
        self.step = step
        self.lr = lr
        self.l2 = l2
        self.lr_dc = lr_dc
        self.lr_dc_step = lr_dc_step
        self.n_node = n_node
        self.max_pos = max_pos
        
        self.build_graph()
        self.reset_parameters()
        self.dropout = nn.Dropout(0.3)
    def build_graph(self):
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=self.step)
        self.pos_embedding = nn.Embedding(self.max_pos, self.hidden_size)
        
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.linear_W = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_dc_step, gamma=self.lr_dc)
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        # hidden: GNN embedding
        batch_size = hidden.shape[0]
        length = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:length]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hidden = hidden + pos_emb

        # v_t: last item vector
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        
        # a: alpha_j * v_j
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        # final scoring
        # # ========================= EDIT HERE ========================
        if not self.nonhybrid:
            item_embeddings = self.embedding.weight
            item_embeddings_norm = item_embeddings / torch.norm(item_embeddings, dim=1).unsqueeze(1)
            a_norm = a / torch.norm(a, dim=1).unsqueeze(1)
            
        
            scores = torch.matmul(a_norm, item_embeddings_norm.transpose(1, 0))
   
        else:
            scores = torch.matmul(a,self.linear_W.weight.transpose(1, 0))
        # ========================= EDIT HERE ========================
          
        scores *= 12 # scaling factor
        return scores

    def forward(self, inputs, A):
        item_embs = self.embedding(inputs)
        
        # ========================= EDIT HERE ========================  
        hidden = F.normalize(item_embs, dim=-1)
    
        # ========================= EDIT HERE ========================  
        hidden = F.normalize(self.gnn(A, hidden),dim=-1)
        
        return hidden
    

class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size

        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)

        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class NISER_session:
    def __init__(self, hidden_size=128, batch_size=128, nonhybrid=False, valid_size=0.1,
                lr=0.001, lr_dc=0.1, lr_dc_step=3, epoch=100, step=1, patience=5, l2=1e-5, device='cuda',
                session_key='SessionId', item_key='ItemId', time_key='Time'):

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nonhybrid = nonhybrid
        self.valid_size = valid_size

        self.lr = lr
        self.lr_dc = lr_dc
        self.lr_dc_step = lr_dc_step
        self.epoch = epoch
        self.step = step
        self.patience = patience
        self.l2 = l2
        self.device = device

        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1

        self.sim_time = 0

    def fit(self, data, test=None, items=None):
        '''
        Trains the predictor.
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''

        # make new session ids(1 ~ #sessions)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.sessionidmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': self.sessionidmap[sessionids].values}), on=self.session_key, how='inner')

        # make new item ids(1 ~ #items)
        itemids = data[self.item_key].unique()
        self.num_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(1, self.num_items+1), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        # sort by time
        data = data.sort_values(by=[self.session_key, self.time_key])

        # make train & valid data
        tr_seqs, tr_labs = [], []
        va_seqs, va_labs = [], []

        num_valid_session = int(self.n_sessions * self.valid_size)
        num_train_session = self.n_sessions - num_valid_session

        sessionlengthmap = data['SessionIdx'].value_counts(sort=False)
        train_session_count = 0
        for sid, session in tqdm(data.groupby(['SessionIdx']), desc='make gnn train data'):
            slen = sessionlengthmap[sid]
            sessionitems = session.sort_values(['Time'])['ItemIdx'].tolist()  # sorted by time
            if train_session_count < num_train_session:
                for t in range(1, slen):
                    tr_seqs.append(sessionitems[:t])
                    tr_labs.append(sessionitems[t])
                train_session_count += 1
            else:
                for t in range(1, slen):
                    va_seqs.append(sessionitems[:t])
                    va_labs.append(sessionitems[t])

        # make test_data for sr_gnn
        te_seqs, te_labs = [], []
        test = pd.merge(test, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}), on=self.item_key, how='inner')
        sessionlengthmap = test[self.session_key].value_counts(sort=False)
        for sid, session in tqdm(test.groupby([self.session_key]), desc='make gnn test data'):
            slen = sessionlengthmap[sid]
            sessionitems = session.sort_values(['Time'])['ItemIdx'].tolist()  # sorted by time
            for t in range(1, slen):
                te_seqs.append(sessionitems[:t])
                te_labs.append(sessionitems[t])

        train_data = (tr_seqs, tr_labs)
        valid_data = (va_seqs, va_labs)
        test_data = (te_seqs, te_labs)

        train_data = Data(train_data, shuffle=True)
        valid_data = Data(valid_data, shuffle=False)

        n_node = self.num_items + 1
        max_pos = max(max([len(x) for x in train_data.inputs]), max([len(x) for x in valid_data.inputs])) + 1

        self.model = trans_to_cuda(NISER(self.hidden_size, self.batch_size, self.nonhybrid, self.step, self.lr,
                                        self.l2, self.lr_dc, self.lr_dc_step, n_node, max_pos))

        start = time.time()
        best_result = [0, 0]
        best_epoch = 0
        bad_counter = 0

        for epoch in range(self.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            if epoch % self.step == 0:
                hit, mrr = train_test(self.model, train_data, valid_data)

                if hit > best_result[0]:
                    best_result[0], best_result[1] = hit, mrr
                    best_epoch = epoch
                    bad_counter = 0
                    torch.save(self.model.state_dict(), f"saves/{self.__class__.__name__}_best_model.pt")
                else:
                    bad_counter += 1

                print('Cur Result:')
                print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d' % (hit, mrr, epoch))
                print('Best Result:')
                print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d' % (best_result[0], best_result[1], best_epoch))
                if bad_counter >= self.patience:
                    break
        print('-------------------------------------------------------')
        end = time.time()
        print("Run time: %f s" % (end - start))

        test_data = Data(test_data, shuffle=False)
        self.test_data = test_data
        
        self.model.load_state_dict(torch.load(f"saves/{self.__class__.__name__}_best_model.pt"))
        self.model.eval()
        
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, timestamp=0, skip=False, type='view'):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        '''
        # new session
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
            self.session_times = []

        if isinstance(input_item_id, list):
            self.session_items = input_item_id
        else:
            if type == 'view':
                self.session_items.append(input_item_id)
                self.session_times.append(timestamp)

        # item id transfomration
        session_items_new_id = self.itemidmap[self.session_items].values
        predict_for_item_ids_new_id = self.itemidmap[predict_for_item_ids].values

        if skip:
            return

        session_items_new_id = session_items_new_id[-self.test_data.len_max:]
        sess_len = len(session_items_new_id)
        inputs = np.zeros([1, self.test_data.len_max])
        inputs[0][:sess_len] = session_items_new_id

        masks = np.zeros_like(inputs)
        masks[0][:len(session_items_new_id)] = 1

        alias_inputs, A, items = self.get_slice(inputs)
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(A).float())
        masks = trans_to_cuda(torch.Tensor(masks).long())

        hidden = self.model.forward(items, A)
        def get(i): return hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

        preds = self.model.compute_scores(seq_hidden, masks).detach().cpu().numpy()
        preds = preds.squeeze()
        preds = np.insert(preds, 0, 0)
        preds = preds[predict_for_item_ids_new_id]

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()

        return series

    def get_slice(self, inputs):
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    def get(i): return hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data, train=True):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    if train == True:
        model.train()
        total_loss = 0.0
        slices = train_data.generate_batch(model.batch_size)
        
        for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
            model.optimizer.zero_grad()
            targets, scores = forward(model, i, train_data)
            targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores, targets - 1)
            loss.backward()
            model.optimizer.step()
            total_loss += loss
            
        print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    
    for i in tqdm(slices):
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
            
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
            
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
                
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
        return alias_inputs, A, items, mask, targets