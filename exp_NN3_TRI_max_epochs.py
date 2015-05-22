"""
Experiment for TRI + NN3

Aim: To find the best max_epochs for TRI(k_min = 2, k_max = 4,5) + NN3(1024, 1024, 1024)
max_epochs: [22, 24, ... ,98, 100]
Averaging 20 models

Summary
             epochs      loss
k_min k_max                  
2     4          76  0.421093
      5          86  0.420173

Time: 5:04:31 on i7-4790k 32G MEM GTX660
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

kf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle = True, random_state = 345)
for train_idx, valid_idx in kf:
    break
y_train = yMat[train_idx]
y_valid = yMat[valid_idx]

# [l1, l2, l3, output]
# params: k_min, k_max, epochs

nIter = 20

m = 130
po = .6

epochs = 20
epochs_add = 2
n_add = 40

bs = 64
mm = .97
lr = .01
dim = 1024
ir = .05
ip = .8
ir_out = .05
mcn_out = 3.5

scores = []
param_grid = {'k_min': [2], 'k_max': [4, 5]}
t0 = datetime.now()
for params in ParameterGrid(param_grid):
    k_min, k_max = params['k_min'], params['k_max']
    predAll = [np.zeros(y_valid.shape) for s in range(n_add)]
    for i in range(nIter):
        seed = i + 9198
        R = col_k_ones_matrix(X.shape[1], m, k_min = k_min, k_max = k_max, seed = seed)
        np.random.seed(seed + 33)
        R.data = np.random.choice([1, -1], R.data.size)
        X3 = X * R
        X1 = np.sign(X3) * np.abs(X3) ** po
        X2 = scaler.fit_transform(X1)
        training = DenseDesignMatrix(X = X2[train_idx], y = yMat[train_idx])
        l1 = RectifiedLinear(layer_name='l1', irange = ir, dim = dim, max_col_norm = 1.)
        l2 = RectifiedLinear(layer_name='l2', irange = ir, dim = dim, max_col_norm = 1.)
        l3 = RectifiedLinear(layer_name='l3', irange = ir, dim = dim, max_col_norm = 1.)
        output = Softmax(layer_name='y', n_classes = 9, irange = ir,
                         max_col_norm = mcn_out)
        mdl = MLP([l1, l2, l3, output], nvis = X2.shape[1])
        trainer = sgd.SGD(learning_rate=lr,
                          batch_size=bs,
                          learning_rule=learning_rule.Momentum(mm),
                          cost=Dropout(default_input_include_prob=ip,
                                       default_input_scale=1/ip),
                          termination_criterion=EpochCounter(epochs),seed = seed)
        decay = sgd.LinearDecayOverEpoch(start=2, saturate=20, decay_factor= .1)
        experiment = Train(dataset = training, model=mdl, algorithm=trainer, extensions=[decay])
        experiment.main_loop()
        epochs_current = epochs
        for s in range(n_add):
            trainer = sgd.SGD(learning_rate=lr * .1,
                              batch_size=bs,
                              learning_rule=learning_rule.Momentum(mm),
                              cost=Dropout(default_input_include_prob=ip,
                                           default_input_scale=1/ip),
                              termination_criterion=EpochCounter(epochs_add),seed = seed)
            experiment = Train(dataset = training, model=mdl, algorithm=trainer)
            experiment.main_loop()
            epochs_current += epochs_add
            pred0 = predict(mdl, X2[train_idx].astype(np.float32))
            pred1 = predict(mdl, X2[valid_idx].astype(np.float32))
            predAll[s] += pred1
            scores.append({'k_min':k_min, 'k_max':k_max,
                           'epochs':epochs_current, 'nModels':i + 1, 'seed':seed,
                           'valid':log_loss(y_valid, pred1),
                           'train':log_loss(y_train, pred0),
                           'valid_avg':log_loss(y_valid, predAll[s] / (i + 1))})
            print scores[-1], datetime.now() - t0

df = pd.DataFrame(scores)

if os.path.exists(path_log) is False:
    print 'mkdir', path_log
    os.mkdir(path_log)
    
df.to_csv(path_log + 'exp_NN3_TRI_max_epochs.csv')
keys = ['k_min', 'k_max', 'epochs']
grouped = df.groupby(keys)

print 'Best'
print pd.DataFrame({'epochs':grouped['valid_avg'].last().unstack().idxmin(1),
                    'loss':grouped['valid_avg'].last().unstack().min(1)})
#              epochs      loss
# k_min k_max                  
# 2     4          76  0.421093
#       5          86  0.420173


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
