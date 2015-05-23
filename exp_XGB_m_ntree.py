"""
Experiment for XGBoost + RI

Aim: To find the best m and ntree
m: [100, 120, 140, 160]
ntree: [140, 160, 180, 200, 220, 240, 260]
Averaging 20 models

Summary

Time: 5:04:31 on i7-4790k 32G MEM GTX660
"""
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import log_loss
from datetime import datetime

import xgboost as xgb
import utility

path = os.getcwd() + '/'
path_log = path + 'logs/'
file_train = path + 'train.csv'

training = pd.read_csv(file_train, index_col = 0)
num_train = training.shape[0]
y = training['target'].values
yMat = pd.get_dummies(training['target']).values
X = training.iloc[:,:93].values
X1 = X / X.mean(0)

kf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle = True, random_state = 345)
for train_idx, valid_idx in kf:
    break

y_train = yMat[train_idx]
y_valid = yMat[valid_idx]

#
nIter = 20

# RI
k = 2

# num_round
nt = 260
nt_lst = [140, 160, 180, 200, 220, 240, 260]
nt_len = len(nt_lst)

# max_depth
tc = 15 

# colsample_bytree
cs = 50. / X.shape[1]

# min_child_weight
mb = 10

# eta
sh = .1

# subsample
bf = .8

pred234 = []
scores = []
t0 = datetime.now()
for m in [100, 120, 140, 160]:
    predAll_train = [np.zeros(y_train.shape) for i in range(nt_len)]
    predAll_valid = [np.zeros(y_valid.shape) for i in range(nt_len)]
    for i in range(nIter):
        seed = i + 12398
        X3 = RI(X1, m, k, normalize = False, seed = seed)
        dtrain , dvalid= xgb.DMatrix(X3[train_idx], label = y), xgb.DMatrix(X3[valid_idx])
        param = {'bst:max_depth':tc, 'bst:eta':sh, 'objective':'multi:softprob','num_class':9,
                 'min_child_weight':mb, 'subsample':bf, 'colsample_bytree':cs,
                 'nthread':8, 'seed':seed, 'silent':1}
        plst = param.items()
        bst = xgb.train(plst, dtrain, nt)
        for j in range(nt_len):
            ntree = nt_lst[j]
            pred_train = bst.predict(dtrain, ntree_limit = ntree).reshape(y_train.shape)
            pred_valid = bst.predict(dvalid, ntree_limit = ntree).reshape(y_valid.shape)
            predAll_train[j] += pred_train
            predAll_valid[j] += pred_valid
            scores.append({'m':m, 'ntree':ntree, 'nModels': i + 1,
                           'train':log_loss(y_train, pred_train),
                           'valid':log_loss(y_valid, pred_valid),
                           'train_avg':log_loss(y_train, predAll_train[j] / (i + 1)),
                           'valid_avg':log_loss(y_valid, predAll_valid[j] / (i + 1))})
        print scores[-1], datetime.now() - t0
    pred234.append(predAll_valid)

r234 = pd.DataFrame(scores)
r234.to_csv(path + 'log/r234.csv')

keys = ['m', 'ntree']

grouped = r234.groupby(keys)

pd.DataFrame({'ntree':grouped['valid_avg'].last().unstack().idxmin(1),
              'loss':grouped['valid_avg'].last().unstack().min(1)})
