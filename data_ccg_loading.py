import pickle, os, pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import interpolate
from sklearn.preprocessing import StandardScaler


def read_data(session, topN, only_positive_lag=False):
    dataset = session[:6]
    ccgs = pickle.load(open(f"./data/{session}/pairs.pkl",'rb'))
    pairs_info = pickle.load(open('./data/allpairs_light.pkl','rb')) 
    pairs_info = pairs_info[pairs_info.dataset==dataset]
    CV_info = pickle.load(open(f'./data/{session}/analysis/CVspikes.pkl','rb')) 
    # # neurons_info = pickle.load(open(f'./data/{session}/analysis/neurons.pkl','rb')) # load using dyad env
    # # neurons_info.to_json(f"./data/{session}/analysis/neurons_info.json")
    # neurons_info = pd.read_json(f"./data/{session}/analysis/neurons_info.json")
    # neurons_info = pd.merge(neurons_info, CV_info, left_index=True, right_index=True)
    print(len(ccgs), np.sum(ccgs.connected | ccgs.inhibitory))

    # create a connection matrix
    _N = CV_info.index.max()+1
    conn_mat = np.zeros((_N,_N), dtype='int')
    conn_mat[ccgs.postIdx, ccgs.preIdx] = ccgs.connected.astype('int') - ccgs.inhibitory.astype('int')
    ccgs['forward'] = conn_mat[ccgs.postIdx, ccgs.preIdx]
    ccgs['backward'] = conn_mat[ccgs.preIdx, ccgs.postIdx]
    labels_np = ccgs.forward.to_numpy() # -1: in, 0: not ex connected, 1: ex
    labels_back_np = ccgs.backward.to_numpy()
    scoresE = pairs_info.scoreE.to_numpy()
    scoresI = pairs_info.scoreI.to_numpy()
    scores = np.max([scoresE, scoresI], axis=0)
    ## preprocessing ccgs
    # collect data
    ccgs_np = np.array([row.ccgs for ri, row in ccgs.iterrows()])  # slow
    lag_ts = np.round(np.arange(-20., 20.1, 0.1), 1) # ms
    # consider only lags of [-10., 10.]
    ccgs_np = ccgs_np[:, (lag_ts>=-10.)&(lag_ts<=10.)]
    lag_ts = lag_ts[(lag_ts>=-10.)&(lag_ts<=10.)]
    # interpolate between -0.3, 0.3
    x = np.array([-0.3, 0.3])
    xnew = np.round(np.arange(-0.3, 0.4, 0.1), 1)
    y = ccgs_np[:, np.isin(lag_ts, x)]
    f = interpolate.interp1d(x, y)
    ccgs_np[:, (lag_ts>=-0.3)&(lag_ts<=0.3)] = f(xnew)
    # consider only positive lags
    if only_positive_lag:
        ccgs_np = ccgs_np[:, lag_ts>=0.]
        lag_ts = lag_ts[lag_ts>=0.]
    else: pass
    # standardize across lag_ts
    ccgs_z = StandardScaler().fit_transform(ccgs_np.T).T

    # get the topN
    nis_incmask = scores>=np.sort(scores)[-topN]
    nis_incmask[ccgs.preIdx==ccgs.postIdx] = False # remove ACGs
    ccgs_z = ccgs_z[nis_incmask]
    labels_np = labels_np[nis_incmask]
    labels_back_np = labels_back_np[nis_incmask]
    scores = scores[nis_incmask]
    
    return ccgs_z, labels_np, labels_back_np, lag_ts, pairs_info[nis_incmask], scores


def read_data_sim(session, topN, randN=None):
    ccgs = pd.read_json(open(f"/Users/wsq/Desktop/proj/synapse_infer/script/data/{session}/pairs.json",'rb'))
    print(len(ccgs), np.sum(ccgs.connected)) # 399368 16778

    # TODO
    labels_np = ccgs.connected.to_numpy()
    isI_np = ccgs.inhibitory.to_numpy() # True: inh, False: exc
    scoresE = ccgs.scoreE.to_numpy()
    scoresI = ccgs.scoreI.to_numpy()
    scores = np.max([scoresE, scoresI], axis=0)
    ## preprocessing ccgs
    # collect data
    ccgs_np = np.array([row.ccgs for ri, row in ccgs.iterrows()]) 
    lag_ts = np.round(np.arange(-10., 10.1, 0.1), 1) # ms
    # no need to # interpolate between -0.3, 0.3
    # standardize across lag_ts
    ccgs_z = StandardScaler().fit_transform(ccgs_np.T).T

    # get the topN
    if randN is None:
        nis_incmask = scores>=np.sort(scores)[-topN]
    else:
        np.random.seed(0)
        nis_incmask = np.zeros(len(ccgs), dtype=bool)
        _randN = np.random.choice(len(ccgs), size=randN, replace=False)
        _goodN = _randN[scores[_randN]>=np.sort(scores[_randN])[-topN]]
        nis_incmask[_goodN] = True
    # remove ACGs
    nis_incmask[ccgs.preIdx==ccgs.postIdx] = False 
    # nis_incmask[labels_np] = True # consider all the connected pairs
    ccgs_z = ccgs_z[nis_incmask]
    print(f"{np.sum(labels_np)} - {np.sum(labels_np[nis_incmask])}") # 16778 - 1653 if randN=4000; 16778 - 10402 if randN=None
    labels_np = labels_np[nis_incmask]
    scores = scores[nis_incmask]
    isI_np = isI_np[nis_incmask]
    
    return ccgs_z, labels_np, isI_np, lag_ts, ccgs[nis_incmask], scores

def read_multiple_data(sessions, topN, sim=False, randN=None, 
                       only_positive_lag=False):
    ccgs_z_all = []; labels_np_all = []; scores_all = []; labels_back_np_all = []; 
    pair_info_all = []
    for sess in tqdm(sessions, "read_multiple_data"):
        if sim:
            data_readed = read_data_sim(sess, topN, randN=randN)
        else:
            data_readed = read_data(sess, topN,
                                    only_positive_lag=only_positive_lag)
        ccgs_z, labels_np, labels_back_np, lag_ts, pair_info, scores = data_readed 
        ccgs_z_all.append(ccgs_z)
        labels_np_all.append(labels_np)
        scores_all.append(scores)
        labels_back_np_all.append(labels_back_np)
        pair_info_all.append(pair_info)
        
    ccgs_z_all = np.concatenate(ccgs_z_all, axis=0)
    labels_np_all = np.concatenate(labels_np_all, axis=0)
    scores_all = np.concatenate(scores_all, axis=0)
    labels_back_np_all = np.concatenate(labels_back_np_all, axis=0)
    pair_info_all = pd.concat(pair_info_all, keys=sessions)
    return ccgs_z_all, labels_np_all, labels_back_np_all, lag_ts, pair_info_all, scores_all

def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def remove_space(s):
    s = s.replace(" ","")
    s = s.replace("'", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace(":", "")
    s = s.replace(",", "_")
    return s

# typically is dict of {area: {eid: xx}}
def load_or_save_dict(_fname, _main, **params):
    if os.path.isfile(_fname):
        with open(_fname, 'rb') as f:
            res = pickle.load(f)
    else:
        res = _main(**params)
        with open(_fname, 'wb') as f:
            pickle.dump(res, f)
    return res

def log_kv(**kwargs):
    print(f"{kwargs}")


if __name__ == '__main__':
    topN = 1000 # 2000
    only_positive_lag = True
    sessions = ["L1R0-1_2022.03.28.01_expt20220503", "L1R0-2_2022.03.28.03_expt20220504", 
                "L1R0-3_2022.04.21.01_expt20220506",]
    data_readed = read_multiple_data(sessions, topN, only_positive_lag=only_positive_lag)
    ccgs_z_all, labels_np_all, labels_back_np_all, lag_ts, pair_info_all, scores_all = data_readed
