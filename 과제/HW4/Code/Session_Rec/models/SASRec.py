import datetime
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")


class SASRec(torch.nn.Module):
    def __init__(self, num_items, hidden_dim, maxlen, num_blocks, num_heads, lr, batch_size, device):
        super().__init__()

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.maxlen = maxlen
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.lr = lr
        self.batch_size = batch_size

        self.device = device

        self.build_graph()

    def build_graph(self):
        self.item_emb = torch.nn.Embedding(self.num_items + 1, self.hidden_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_dim)
        self.emb_dropout = torch.nn.Dropout(p=0.2)
        
        # Attention Layer
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        
        # Position-wise Feed-Forward Layer
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_dim)

        # Self attention block
        for _ in range(self.num_blocks):
            # layer normalization layer (self-attention)
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_dim)
            self.attention_layernorms.append(new_attn_layernorm)

            # self-attention layer
            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_dim, self.num_heads, 0.2, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            # layer norm layer (position-wise feed-forward)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_dim)
            self.forward_layernorms.append(new_fwd_layernorm)

            # position-wise feed-forward layer
            new_fwd_layer = PointWiseFeedForward(self.hidden_dim, 0.2)
            self.forward_layers.append(new_fwd_layer)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)

    def forward(self, inputs):
        # inputs: (batch_size, maxlen)
        seqs = self.item_emb(inputs)
        
        # Item Embedding + Positional Embedding
        # ========================= EDIT HERE ========================
        positions = torch.arange(inputs.shape[1], device=self.device).unsqueeze(0)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        # ========================= EDIT HERE ========================  
        
        timeline_mask = (inputs == 0)
        seqs = seqs * ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        # Self-attention block
        for i in range(len(self.attention_layers)):
            # ========================= EDIT HERE ========================  
            # layer normalization
            seqs = self.attention_layernorms[i](seqs)
            # self-attention
            attn_output, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
            # residual connection
            seqs = seqs + attn_output
            # layer normalization
            seqs = self.forward_layernorms[i](seqs)
            # position-wise feed-forward
            seqs = self.forward_layers[i](seqs)
            
            # ========================= EDIT HERE ========================  
            seqs = seqs * ~timeline_mask.unsqueeze(-1)

        # (batch, maxlen, hidden_dim)
        hidden_feats = self.last_layernorm(seqs)

        final_feat = hidden_feats[:, -1, :]
        item_embs = self.item_emb.weight[1:].T
        logits = final_feat @ item_embs
        
        return logits

    def train_model_per_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets - 1)
        loss.backward()
        self.optimizer.step()

        return loss

    def restore(self):
        with open(f"saves/{self.__class__.__name__}_best_model.pt", 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)


class SASRec_session:
    def __init__(self, hidden_dim=50, batch_size=128, valid_size=0.1, lr=0.001, num_epochs=100, num_blocks=2, num_heads=1, maxlen=50,
                step=1, patience=5, device='cuda', session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.valid_size = valid_size

        self.lr = lr
        self.num_epochs = num_epochs
        self.maxlen = maxlen
        self.step = step
        self.patience = patience
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

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
        self.itemidmap = pd.Series(data=np.arange(1, self.num_items + 1), index=itemids)
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
        for sid, session in tqdm(data.groupby(['SessionIdx']), desc='make train data'):
            slen = sessionlengthmap[sid]
            sessionitems = session.sort_values(['Time'])['ItemIdx'].tolist()  # sorted by time
            if train_session_count < num_train_session:
                for t in range(1, slen):
                    tr_seqs.append(sessionitems[0 if t < self.maxlen else t - self.maxlen:t])
                    tr_labs.append(sessionitems[t])
                train_session_count += 1
            else:
                for t in range(1, slen):
                    va_seqs.append(sessionitems[0 if t < self.maxlen else t - self.maxlen:t])
                    va_labs.append(sessionitems[t])

        train_data = (tr_seqs, tr_labs)
        valid_data = (va_seqs, va_labs)
        
        train_data = Data(train_data)
        valid_data = Data(valid_data)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.model = SASRec(self.num_items, self.hidden_dim, self.maxlen, self.num_blocks, self.num_heads, self.lr,
                            self.batch_size, self.device).to(self.device)
        
        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0

        for epoch in range(self.num_epochs):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            if epoch % self.step == 0:
                hit, mrr = self.train_test(train_loader, valid_loader)

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
        
        self.model.load_state_dict(torch.load(f"saves/{self.__class__.__name__}_best_model.pt"))
        self.model.eval()
    
    def train_test(self, train_loader, test_loader, train=True):
        if train == True:
            self.model.train()
            total_loss = 0.0
            
            for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train', dynamic_ncols=True):
                seq, target = seq.to(self.device), target.to(self.device)
                loss = self.model.train_model_per_batch(seq, target)
                total_loss += loss
                
            print('\tLoss:\t%.3f' % total_loss)

        print('start predicting: ', datetime.datetime.now())
        self.model.eval()
        hit, mrr = [], []
        
        for i, (seqs, targets, lens) in tqdm(enumerate(test_loader), total=len(test_loader), desc='test', dynamic_ncols=True):
            seqs, targets = seqs.to(self.device), targets.to(self.device)
            scores = self.model(seqs)
            sub_scores = scores.topk(20)[1].detach().cpu().numpy()
            
            for score, target in zip(sub_scores, targets):
                target = target.detach().cpu().numpy()
                hit.append(np.isin(target - 1, score))
                
                if len(np.where(score == (target - 1))[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == (target - 1))[0][0] + 1))
                    
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        
        return hit, mrr
        
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
            
        seqlen = len(session_items_new_id)
        if seqlen < self.maxlen:
            seq = np.concatenate((session_items_new_id, np.array([0] * (self.maxlen - seqlen))))
        else:
            seq = session_items_new_id[-self.maxlen:]
        
        seq = seq.reshape(1, -1)
        with torch.no_grad():
            seq = torch.from_numpy(seq).long().to(self.device)
            scores = self.model.forward(seq)

        preds = scores.view(-1).cpu().detach().numpy()
        preds = preds[predict_for_item_ids_new_id - 1]

        series = pd.Series(data=preds, index=predict_for_item_ids)
        series = series / series.max()
        
        return series


def collate_fn(data):
    """
    This function will be used to pad the sessions to max length
    in the batch and transpose the batch from 
    batch_size x max_seq_len to max_seq_len x batch_size.
    It will return padded vectors, labels and lengths of each session (before padding)
    It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), 50).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i, :lens[i]] = torch.LongTensor(sess)
        labels.append(label)

    return padded_sesss, torch.tensor(labels).long(), lens


class Data(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """

    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)

    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])
    

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
