"""
Experiment for NN4(RI)

Aim: To find the best max_epochs for NN4(*, 1024, 1024, 1024) + RI(k = 3, m = 200)
max_epochs: [22, 24, ... ,98, 140]
Averaging 20 models

Summary
epochs 88 , loss 0.421860471364
Time:3:40:30  on i7-4790k 32G MEM GTX660

I got a different result, epochs 112 loss 0.422868, before I reinstalled ubuntu 14.04 LTS.
So I chose max_epochs = 112.
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

# Params for RI
m = 200
k = 3

# Params for NN
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
        pred_train = predict(mdl, X2[train_idx].astype(np.float32))
        pred_valid = predict(mdl, X2[valid_idx].astype(np.float32))
        predAll[s] += pred_valid
        scores.append({'epochs':epochs_current, 'nModels':i + 1, 'seed':seed,
                       'train':log_loss(y_train, pred_train),
                       'valid':log_loss(y_valid, pred_valid),
                       'valid_avg':log_loss(y_valid, predAll[s] / (i + 1))})
        print scores[-1], datetime.now() - t0

df = pd.DataFrame(scores)

if os.path.exists(path_log) is False:
    print 'mkdir', path_log
    os.mkdir(path_log)

df.to_csv(path_log + 'exp_NN4_RI_max_epochs.csv')
keys = ['epochs']

grouped = df.groupby(keys)

print 'epochs',grouped['valid_avg'].last().idxmin(),', loss',grouped['valid_avg'].last().min()
# epochs 88 , loss 0.421860471364

g = grouped[['train', 'valid']].mean()
g['valid_avg'] = grouped['valid_avg'].last()

print g.iloc[[0,1,32,33,34,58,59],:]
#            train     valid  valid_avg
# epochs                               
# 22      0.319737  0.468458   0.436766
# 24      0.313538  0.468300   0.435694
# 86      0.193640  0.486078   0.422321
# 88      0.190694  0.487625   0.421860
# 90      0.187374  0.487897   0.421998
# 138     0.134388  0.512527   0.423662
# 140     0.132642  0.514666   0.425003

ax = g.plot()
ax.set_title('NN4(RI) m=200, k=3')
ax.set_ylabel('Logloss')
fig = ax.get_figure()
fig.savefig(path_log + 'exp_NN4_RI_max_epochs.png')
