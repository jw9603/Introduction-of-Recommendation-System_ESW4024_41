import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data_session(data_name='music'):
    if data_name == 'music':
        train_path = f'./data/train.tsv'
        test_path = f'./data/test.tsv'
        train_df = pd.read_csv(train_path, sep='\t') 
        test_df = pd.read_csv(test_path, sep='\t')
    elif data_name == 'kaggle':
        # kaggle data
        train_path = f'./data/train_data.csv'
        test_path = f'./data/test_data.csv'
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
    return train_df, test_df


def eval_session(model, train_df, test_df, top_k=20):
    session_key='SessionId'
    item_key='ItemId'
    time_key='Time'

    actions=len(test_df)
    sessions=len(test_df[session_key].unique())
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    items_to_predict=train_df[item_key].unique()
    prev_iid, prev_sid=-1, -1

    recall_list=[]
    mrr_list=[]

    for i in tqdm(range(actions), desc="Eval...", dynamic_ncols=True):
        # Get sid, iid, ts of current row
        sid=test_df[session_key].values[i]
        iid=test_df[item_key].values[i]
        ts=test_df[time_key].values[i]

        # if new session
        if prev_sid != sid:
            prev_sid = sid
        else:
            preds = model.predict_next(sid, prev_iid, items_to_predict, timestamp=ts)
            preds[np.isnan(preds)] = 0
            preds.sort_values(ascending=False, inplace=True)

            # Get top_k items
            top_k_preds=preds.iloc[:top_k]
            if iid in top_k_preds.index:
                rank= 1/(top_k_preds.index.get_loc(iid) + 1)
                hit=1
            else:
                rank, hit=0, 0
            recall_list.append(hit)
            mrr_list.append(rank)
        prev_iid = iid
    recall=np.mean(recall_list)
    mrr=np.mean(mrr_list)
    return recall, mrr


def save_submission(model, train_df, test_df, top_k=20):
    session_key='SessionId'
    item_key='ItemId'
    time_key='Time'

    actions=len(test_df)
    sessions=len(test_df[session_key].unique())
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    items_to_predict=train_df[item_key].unique()
    prev_sid = -1
    prev_iids = []

    submission = pd.DataFrame(columns=['SessionId', 'ItemId'])
    submission['SessionId'] = test_df['SessionId'].unique()
    
    for i in tqdm(range(actions), desc="Eval...", dynamic_ncols=True):
        # Get sid, iid, ts of current row
        sid=test_df[session_key].values[i]
        iid=test_df[item_key].values[i]
        ts=test_df[time_key].values[i]

        # if new session and prev_iids is not empty
        if prev_sid != sid and len(prev_iids) > 0:
            preds = model.predict_next(sid, prev_iids, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds.sort_values(ascending=False, inplace=True)

            # Get top_k items
            top_k_preds = preds.iloc[:top_k]
            submission.loc[submission['SessionId'] == prev_sid, 'ItemId'] = ' '.join(top_k_preds.index.astype('str').values)

        # if new session
        if prev_sid != sid:
            prev_sid = sid
            prev_iids = [iid]
        else:
            prev_iids.append(iid)

        # inference of last session
        if i == actions - 1:
            preds = model.predict_next(sid, prev_iids, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds.sort_values(ascending=False, inplace=True)

            # Get top_k items
            top_k_preds = preds.iloc[:top_k]
            submission.loc[submission['SessionId'] == prev_sid, 'ItemId'] = ' '.join(top_k_preds.index.astype('str').values)

    submission.to_csv(f'submission_{model.__class__.__name__}_best.csv', index=False)
