"""
Experiment for XGBoost + RI

Aim: To find the best m and ntree(num_round)
m: [100, 120, 140, 160]
ntree: [140, 160, 180, 200, 220, 240, 260]
Averaging 20 models

Summary
         loss  ntree
m                   
100  0.450670    240
120  0.450491    220
140  0.449575    220
160  0.449249    220 *

Time: 2:56:52 on i7-4790k 32G MEM GTX660

I got a different result before I reinstalled ubuntu 14.04 LTS.
         loss  ntree
m                   
100  0.450663    240
120  0.449751    220
140  0.448961    220 *
160  0.449046    220

So I chose m=140, ntree=220.

"""
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from datetime import datetime
import os

import xgboost as xgb

from utility import *

path = os.getcwd() + '/'
path_log = path + 'logs/'
file_train = path + 'train.csv'

training = pd.read_csv(file_train, index_col = 0)
num_train = training.shape[0]
y = training['target'].values
yMat = pd.get_dummies(training['target']).values
X = training.iloc[:,:93].values
X1 = X / X.mean(0)

kf = StratifiedKFold(y, n_folds=5, shuffle = True, random_state = 345)
for train_idx, valid_idx in kf:
    break

y_train_1 = yMat[train_idx].argmax(1)
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

scores = []
t0 = datetime.now()
for m in [100, 120, 140, 160]:
    predAll_train = [np.zeros(y_train.shape) for i in range(nt_len)]
    predAll_valid = [np.zeros(y_valid.shape) for i in range(nt_len)]
    for i in range(nIter):
        seed = i + 12398
        X3 = RI(X1, m, k, normalize = False, seed = seed)
        dtrain , dvalid= xgb.DMatrix(X3[train_idx], label = y_train_1), xgb.DMatrix(X3[valid_idx])
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
            scores.append({'m':m, 'ntree':ntree, 'nModels': i + 1, 'seed':seed,
                           'train':log_loss(y_train, pred_train),
                           'valid':log_loss(y_valid, pred_valid),
                           'train_avg':log_loss(y_train, predAll_train[j] / (i + 1)),
                           'valid_avg':log_loss(y_valid, predAll_valid[j] / (i + 1))})
        print scores[-1], datetime.now() - t0

df = pd.DataFrame(scores)

if os.path.exists(path_log) is False:
    print 'mkdir', path_log
    os.mkdir(path_log)

df.to_csv(path_log + 'exp_XGB_RI_m_ntree.csv')

keys = ['m', 'ntree']

grouped = df.groupby(keys)

print pd.DataFrame({'ntree':grouped['valid_avg'].last().unstack().idxmin(1),
                    'loss':grouped['valid_avg'].last().unstack().min(1)})
#          loss  ntree
# m                   
# 100  0.450670    240
# 120  0.450491    220
# 140  0.449575    220
# 160  0.449249    220

#
grouped = df[df['m'] == 140].groupby('ntree')
g = grouped[['valid']].mean()
g['valid_avg'] = grouped['valid_avg'].last()
print g
#           valid  valid_avg
# ntree                     
# 140    0.477779   0.454885
# 160    0.476271   0.452038
# 180    0.476112   0.450559
# 200    0.476564   0.449759
# 220    0.477543   0.449575
# 240    0.478995   0.449745
# 260    0.480710   0.450266

ax = g.plot()
ax.set_title('XGB+RI k=2, m=140')
ax.set_ylabel('Logloss')
fig = ax.get_figure()
fig.savefig(path_log + 'exp_XGB_RI_m_ntree.png')
