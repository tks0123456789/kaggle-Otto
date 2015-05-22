"""
Experiment for NN4 + RI

Aim: To find the best m and max_epochs for NN4(*, 1024, 1024, 1024) + RI(k = 3, m = m)
m: [160, 200, 240]
max_epochs: [22, 24, ... ,98, 100]
Averaging 20 models

Summary
     epochs      loss
 m                    
160     126  0.423487
200     112  0.422868 **
240      92  0.425127

Time:  on i7-4790k 32G MEM GTX660
"""
import numpy as np
import scipy as sp
import pandas as pd
from pylearn2.models import mlp
from pylearn2.models.mlp import RectifiedLinear, Softmax, MLP
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.train import Train

from theano.compat.python2x import OrderedDict
import theano.tensor as T
from theano import function

import pickle
import sklearn.preprocessing as pp
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.metrics import log_loss
from sklearn.grid_search import ParameterGrid

from datetime import datetime

import os
from utility import *
from predict import predict

import pylab

path = os.getcwd() + '/'
path_log = path + 'logs/'
file_train = path + 'train.csv'

training = pd.read_csv(file_train, index_col = 0)
num_train = training.shape[0]
y = training['target'].values
yMat = pd.get_dummies(training['target']).values
X = training.iloc[:,:93].values
scaler = pp.StandardScaler()
X2 = scaler.fit_transform(X ** .6)


kf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle = True, random_state = 345)
for train_idx, valid_idx in kf:
    break
y_train = yMat[train_idx]
y_valid = yMat[valid_idx]

training = DenseDesignMatrix(X = X2[train_idx], y = y_train)
valid = DenseDesignMatrix(X = X2[valid_idx], y = y_valid)

# [l1, l2, l3, l4, output]
nIter = 20

k = 3

epochs = 20
epochs_add = 2
n_add = 60

bs = 64
mm = .97
lr = .01
dim2 = 1024
ir1 = .01
ir2 = .05
ip = .8
ir_out = .05
mcn_out = 2.5

scores = []
t0 = datetime.now()
for m in [160, 200, 240]:
    predAll = [np.zeros(y_valid.shape) for s in range(n_add)]
    for i in range(nIter):
        seed = i + 3819
        R = RImatrix(X.shape[1], m, k, rm_dup_cols = True, seed = seed)
        R = np.abs(R.todense().astype(np.float32))
        dim1 = R.shape[1]
        l1 = RectifiedLinear(layer_name='l1', irange = ir1, dim = dim1, mask_weights = R)
        l2 = RectifiedLinear(layer_name='l2', irange = ir2, dim = dim2, max_col_norm = 1.)
        l3 = RectifiedLinear(layer_name='l3', irange = ir2, dim = dim2, max_col_norm = 1.)
        l4 = RectifiedLinear(layer_name='l4', irange = ir2, dim = dim2, max_col_norm = 1.)
        output = Softmax(layer_name='y', n_classes = 9, irange = ir_out,
                          max_col_norm = mcn_out)
        mdl = MLP([l1, l2, l3, l4, output], nvis = X2.shape[1])
        trainer = sgd.SGD(learning_rate=lr,
                          batch_size=bs,
                          learning_rule=learning_rule.Momentum(mm),
                          cost=Dropout(input_include_probs = {'l1':1.},
                                       input_scales = {'l1':1.},
                                       default_input_include_prob=ip,
                                       default_input_scale=1/ip),
                          termination_criterion=EpochCounter(epochs),seed = seed)
        decay = sgd.LinearDecayOverEpoch(start=2, saturate=20, decay_factor= .1)
        experiment = Train(dataset = training, model=mdl, algorithm=trainer, extensions=[decay])
        experiment.main_loop()
        epochs_current = epochs
        for s in range(n_add):
            del mdl.monitor
            trainer = sgd.SGD(learning_rate=lr * .1,
                              batch_size=bs,
                              learning_rule=learning_rule.Momentum(mm),
                              cost=Dropout(input_include_probs = {'l1':1.},
                                           input_scales = {'l1':1.},
                                           default_input_include_prob=ip,
                                           default_input_scale=1/ip),
                              termination_criterion=EpochCounter(epochs_add),seed = seed)
            experiment = Train(dataset = training, model=mdl, algorithm=trainer)
            experiment.main_loop()
            epochs_current += epochs_add
            pred0 = predict(mdl, X2[idx[0],:].astype(np.float32))
            pred1 = predict(mdl, X2[idx[1],:].astype(np.float32))
            predAll[s] += pred1
            scores.append({'m':m,
                           'epochs':epochs_current, 'nModels':i + 1, 'seed':seed,
                           'valid':log_loss(y_valid, pred1),
                           'train':log_loss(y_train, pred0),
                           'avg':log_loss(y_valid, predAll[s] / (i + 1))})
            print scores[-1], datetime.now() - t0

df = pd.DataFrame(scores)
df.to_csv(path + 'log/exp_NN4_RI_m_max_epochs.csv')
keys = ['m', 'epochs']

grouped = df.groupby(keys)

print 'Best'
print pd.DataFrame({'epochs':grouped['avg'].last().unstack().idxmin(1),
                    'loss':grouped['avg'].last().unstack().min(1)})
#      epochs      loss
# m                    
# 160     126  0.423487
# 200     112  0.422868 **
# 240      92  0.425127


print grouped[['valid', 'train']].mean().unstack().iloc[:,[45,52, 55, 105, 112, 115]]



"""
# Figure for k_max == 4
grouped = df[df['k_max'] == 4].groupby('epochs')
g = grouped[['train', 'valid']].mean()
g['valid_avg'] = grouped['valid_avg'].last()
print g.iloc[[0,1,26,27,28,38,39],:]
#            train     valid  valid_avg
# epochs                               
# 22      0.280855  0.478790   0.431065
# 24      0.274300  0.479380   0.430083
# 74      0.173661  0.504325   0.422263
# 76      0.170654  0.505458   0.421093
# 78      0.167444  0.506752   0.421296
# 98      0.142868  0.519850   0.422619
# 100     0.140718  0.521398   0.422675

ax = g.plot()
ax.set_title('TRI+NN3 k_min=2, k_max=4')
ax.set_ylabel('Logloss')
fig = ax.get_figure()
fig.savefig(path_log + 'exp_NN3_TRI_max_epochs.png')
"""
quit()
