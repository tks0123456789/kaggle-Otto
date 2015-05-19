"""
XGBoost + Count feature, Time: 1:37:22
XGBoost + Random indexing, Time: 1:56:58
The running times on i7 4790k, 32G MEM, GTX660
"""

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.metrics import log_loss
import xgboost as xgb
from datetime import datetime

from utility import *

path = './'
file_train = path + 'train.csv'
file_test = path + 'test.csv'

training = pd.read_csv(file_train, index_col = 0)
test = pd.read_csv(file_test, index_col = 0)
num_train = training.shape[0]

target = training['target']
training.drop('target', inplace = True, axis = 1)

df_trans = pd.Series(range(9), index = target.unique())
y = target.map(df_trans).values
yMat = pd.get_dummies(y).values

X = np.vstack((training.values, test.values))


# Common parameters
nIter = 50
tc = 15 # max_depth
sh = .1 # eta
bf = .8 # subsample

### XGB1: Count feature
nt = 260
mb = 5 # min_child_weight
cs = 45. / 93 # colsample_bytree

X2, ignore = count_feature(X)
dtrain , dtest = xgb.DMatrix(X2[:num_train], label = y), xgb.DMatrix(X2[num_train:])

predAll_train = np.zeros((num_train, 9))
predAll_test = np.zeros((test.shape[0], 9))
scores = []

t0 = datetime.now()
for i in range(nIter):
    seed = i + 123
    param = {'bst:max_depth':tc, 'bst:eta':sh, 'silent':1, 'objective':'multi:softprob','num_class':9,
             'min_child_weight':mb, 'subsample':bf, 'colsample_bytree':cs, 'nthread':8, 'seed':seed}
    plst = param.items()
    bst = xgb.train(plst, dtrain, nt)
    # bst.save_model(path + 'model/model_XGB_CF_' + str(seed) + '.model')
    pred_train = bst.predict(dtrain).reshape((num_train, 9))
    pred_test = bst.predict(dtest).reshape(predAll_test.shape)
    predAll_train += pred_train
    predAll_test += pred_test
    sc1 = log_loss(yMat, pred_train)
    sc2 = log_loss(yMat, predAll_train / (i + 1))
    print i, "  each:%f, avg:%f, Time:%s" % (sc1, sc2, datetime.now() - t0)

# 49   each:0.131026, avg:0.130512, Time:1:37:21.187174
pred_XGB_CF = predAll_test / nIter
np.save(path + 'pred_XGB_CF.npy', pred_XGB_CF)

### XGB2: Random Indexing
X1 = X / X.mean(0)

m = 140
k = 2
nt = 220
mb = 10 # min_child_weight
cs = 50. / 93 # colsample_bytree

predAll_train = np.zeros((num_train, 9))
predAll_test = np.zeros((test.shape[0], 9))
t0 = datetime.now()
for i in range(nIter):
    seed = i + 123210
    X3 = RI(X1, m, k, normalize = False, seed = seed)
    dtrain = xgb.DMatrix(X3[:num_train], label = y)
    dtest = xgb.DMatrix(X3[num_train:])
    param = {'bst:max_depth':tc, 'bst:eta':sh, 'silent':1, 'objective':'multi:softprob',
             'num_class':9, 'min_child_weight':mb, 'subsample':bf,
             'colsample_bytree':cs, 'nthread':8, 'seed':seed}
    plst = param.items()
    bst = xgb.train(plst, dtrain, nt)
    # bst.save_model(path + 'model/model_XGB_RI_' + str(seed) + '.model')
    pred_train = bst.predict(dtrain).reshape((num_train, 9))
    pred_test = bst.predict(dtest).reshape(predAll_test.shape)
    predAll_train += pred_train
    predAll_test += pred_test
    sc1 = log_loss(yMat, pred_train)
    sc2 = log_loss(yMat, predAll_train / (i + 1))
    print i, "  each:%f, avg:%f, Time:%s" % (sc1, sc2, datetime.now() - t0)

# 49   each:0.085345, avg:0.084942, Time:1:56:57.580157
pred_XGB_RI = predAll_test / nIter
np.save(path + 'pred_XGB_RI.npy', pred_XGB_RI)

