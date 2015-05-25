"""
Experiment for XGBoost + CF

Aim: To find the best tc(max_depth), mb(min_child_weight), mf(colsample_bytree * 93), ntree
tc: [13, 15, 17]
mb: [5, 7, 9]
mf: [40, 45, 50, 55, 60]
ntree: [160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360]
Averaging 20 models

Summary

Best
         loss                                 ntree                    
mf         40      45      50      55      60    40   45   50   55   60
tc mb                                                                  
13 5   0.4471  0.4471  0.4473  0.4471  0.4476   300  300  280  280  260
   7   0.4477  0.4475  0.4469  0.4472  0.4481   340  320  300  300  300
   9   0.4485  0.4484  0.4487  0.4488  0.4487   360  360  340  340  340
15 5   0.4471 *0.4465* 0.4471  0.4476  0.4478   260 *260* 240  240  240
   7   0.4473  0.4468  0.4473  0.4474  0.4478   300  280  260  260  260
   9   0.4483  0.4480  0.4483  0.4484  0.4492   340  320  300  300  280
17 5   0.4471  0.4472  0.4474  0.4476  0.4478   240  240  220  220  200
   7   0.4474  0.4470  0.4468  0.4475  0.4473   280  260  260  240  240
   9   0.4481  0.4480  0.4476  0.4480  0.4486   320  300  280  260  260

Time: 1 day, 7:37:21 on i7-4790k 32G MEM GTX660
"""
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from datetime import datetime
import os
from sklearn.grid_search import ParameterGrid

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

kf = StratifiedKFold(y, n_folds=5, shuffle = True, random_state = 345)
for train_idx, valid_idx in kf:
    break

y_train_1 = yMat[train_idx].argmax(1)
y_train = yMat[train_idx]
y_valid = yMat[valid_idx]

X2, ignore = count_feature(X)
dtrain , dvalid= xgb.DMatrix(X2[train_idx], label = y_train_1), xgb.DMatrix(X2[valid_idx])

#
nIter = 20

nt = 360
nt_lst = range(160, 370, 20)
nt_len = len(nt_lst)

bf = .8 # subsample
sh = .1 # eta
# tc:max_depth, mb:min_child_weight, mf(max features):colsample_bytree * 93
param_grid = {'tc':[13, 15, 17], 'mb':[5, 7, 9], 'mf':[40, 45, 50, 55, 60]}

scores = []
t0 = datetime.now()
for params in ParameterGrid(param_grid):
    tc = params['tc']
    mb = params['mb']
    mf = params['mf']
    cs = float(mf) / X.shape[1]
    print tc, mb, mf
    predAll = [np.zeros(y_valid.shape) for k in range(nt_len)]
    for i in range(nIter):
        seed = 112233 + i
        param = {'bst:max_depth':tc, 'bst:eta':sh,'objective':'multi:softprob','num_class':9,
                 'min_child_weight':mb, 'subsample':bf, 'colsample_bytree':cs,
                 'silent':1, 'nthread':8, 'seed':seed}
        plst = param.items()
        bst = xgb.train(plst, dtrain, nt)
        for s in range(nt_len):
            ntree = nt_lst[s]
            pred = bst.predict(dvalid, ntree_limit = ntree).reshape(y_valid.shape)
            predAll[s] += pred
            scores.append({'tc':tc, 'mb':mb, 'mf':mf, 'ntree':ntree, 'nModels':i+1, 'seed':seed,
                           'valid':log_loss(y_valid, pred),
                           'valid_avg':log_loss(y_valid, predAll[s] / (i+1))})
        print scores[-4], datetime.now() - t0
            
df = pd.DataFrame(scores)

if os.path.exists(path_log) is False:
    print 'mkdir', path_log
    os.mkdir(path_log)

df.to_csv(path_log + 'exp_XGB_CF_tc_mb_mf_ntree.csv')

keys = ['tc', 'mb', 'mf', 'ntree']
grouped = df.groupby(keys)

pd.set_option('display.precision', 5)
print pd.DataFrame({'loss':grouped['valid_avg'].last().unstack().min(1),
                    'ntree':grouped['valid_avg'].last().unstack().idxmin(1)}).unstack()
#          loss                                 ntree                    
# mf         40      45      50      55      60    40   45   50   55   60
# tc mb                                                                  
# 13 5   0.4471  0.4471  0.4473  0.4471  0.4476   300  300  280  280  260
#    7   0.4477  0.4475  0.4469  0.4472  0.4481   340  320  300  300  300
#    9   0.4485  0.4484  0.4487  0.4488  0.4487   360  360  340  340  340
# 15 5   0.4471  0.4465  0.4471  0.4476  0.4478   260  260  240  240  240
#    7   0.4473  0.4468  0.4473  0.4474  0.4478   300  280  260  260  260
#    9   0.4483  0.4480  0.4483  0.4484  0.4492   340  320  300  300  280
# 17 5   0.4471  0.4472  0.4474  0.4476  0.4478   240  240  220  220  200
#    7   0.4474  0.4470  0.4468  0.4475  0.4473   280  260  260  240  240
#    9   0.4481  0.4480  0.4476  0.4480  0.4486   320  300  280  260  260

print pd.DataFrame({'loss':grouped['valid'].mean().unstack().min(1),
                    'ntree':grouped['valid'].mean().unstack().idxmin(1)}).unstack()
#          loss                                 ntree                    
# mf         40      45      50      55      60    40   45   50   55   60
# tc mb                                                                  
# 13 5   0.4563  0.4564  0.4564  0.4561  0.4566   280  260  260  260  240
#    7   0.4565  0.4563  0.4557  0.4561  0.4569   320  300  300  300  280
#    9   0.4571  0.4569  0.4571  0.4573  0.4570   340  340  320  300  300
# 15 5   0.4567  0.4559  0.4565  0.4571  0.4571   260  240  240  220  220
#    7   0.4565  0.4558  0.4562  0.4564  0.4568   280  260  260  260  240
#    9   0.4570  0.4567  0.4570  0.4570  0.4577   300  300  280  280  260
# 17 5   0.4568  0.4569  0.4570  0.4572  0.4574   220  220  200  200  200
#    7   0.4567  0.4563  0.4559  0.4567  0.4564   260  240  240  220  220
#    9   0.4571  0.4569  0.4565  0.4567  0.4573   280  280  260  260  240

#
criterion = df.apply(lambda x: x['tc']==15 and x['mb']==5 and x['mf']==45, axis = 1)

grouped = df[criterion].groupby('ntree')
g = grouped[['valid']].mean()
g['valid_avg'] = grouped['valid_avg'].last()
print g
#           valid  valid_avg
# ntree                     
# 160    0.461023   0.452912
# 180    0.458513   0.450111
# 200    0.456939   0.448232
# 220    0.456147   0.447141
# 240    0.455870   0.446598
# 260    0.456097   0.446525
# 280    0.456657   0.446827
# 300    0.457434   0.447327
# 320    0.458462   0.448101
# 340    0.459635   0.449036
# 360    0.460977   0.450160

ax = g.plot()
ax.set_title('XGB+CF max_depth=15\n min_child_weight=5, colsample_bytree=45/93.')
ax.set_ylabel('Logloss')
fig = ax.get_figure()
fig.savefig(path_log + 'exp_XGB_CF_tc_mb_mf_ntree.png')
