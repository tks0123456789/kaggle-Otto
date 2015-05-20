"""
Transposed Random Indexing + NN3(3 hidden layers), Time: 21:42:15
NN4(4 hidden layers, the first is sparse) + Random Indexing, Time: 14:47:57
The running times on i7 4790k, 32G MEM, GTX660
"""

from pylearn2.models.mlp import RectifiedLinear, Softmax, MLP
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.train import Train

import numpy as np
import scipy as sp
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.metrics import log_loss
from datetime import datetime

from utility import *
from predict import predict

path = './'
file_train = path + 'train.csv'
file_test = path + 'test.csv'

training = pd.read_csv(file_train, index_col = 0)
num_train = training.shape[0]
yMat = pd.get_dummies(training['target']).values
test = pd.read_csv(file_test, index_col = 0)
X = np.vstack([training.iloc[:,:93].values, test.values])
scaler = pp.StandardScaler()


# Common parameters
nIter = 100
bs = 64 # batch_size
mm = .97 # momentum
lr = .01 # learning_rate
dim = 1024
ir = .05 # irange
ip = .8 # input_include_prob = 1 - dropout probability


### 3 hidden layers(dense, dense, dense)
### Transposed Random Indexing
m = 130
k_min = 2
po = .6
mcn_out = 3.5 # max_col_norm for the output layer

params_lst = [{'k_max':4, 'epochs':76}, {'k_max':5, 'epochs':88}]

# 21:42:15.127917
t0 = datetime.now()
for params in params_lst:
    k_max = params['k_max']
    epochs = params['epochs']
    print k_max, epochs
    predAll_train = np.zeros((num_train, 9))
    predAll_test = np.zeros((test.shape[0], 9))
    for i in range(nIter):
        seed = i + 987654
        R = col_k_ones_matrix(X.shape[1], m, k_min = k_min, k_max = k_max, seed = seed)
        np.random.seed(seed + 34)
        R.data = np.random.choice([1, -1], R.data.size)
        X3 = X * R
        X1 = np.sign(X3) * np.abs(X3) ** po
        X2 = scaler.fit_transform(X1)
        training = DenseDesignMatrix(X = X2[:num_train], y = yMat)
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
                          termination_criterion=EpochCounter(epochs), seed = seed)
        decay = sgd.LinearDecayOverEpoch(start=2, saturate=20, decay_factor= .1)
        #fname = path + 'model/TRI_' + 'kmax_'+ str(k_max) + '_seed_' + str(seed) + '.pkl'
        experiment = Train(dataset = training, model=mdl, algorithm=trainer, extensions=[decay])
        #                   save_path = fname, save_freq = epochs)
        experiment.main_loop()
        pred_train = predict(mdl, X2[:num_train].astype(np.float32))
        pred_test = predict(mdl, X2[num_train:].astype(np.float32))
        predAll_train += pred_train
        predAll_test += pred_test
        sc1 = log_loss(yMat, pred_train)
        sc2 = log_loss(yMat, predAll_train / (i + 1))
        print i, "  each:%f, avg:%f, Time:%s" % (sc1, sc2, datetime.now() - t0)
    pred_nn = predAll_test / nIter
    np.save(path + 'pred_TRI_kmax_' + str(k_max) + '.npy', pred_nn)


### 4 hidden layers(sparse, dense, dense, dense)
### RI defines the sparse connections.
X2 = scaler.fit_transform(X ** .6)
X_train = X2[:num_train].astype(np.float32)
X_test = X2[num_train:].astype(np.float32)
training = DenseDesignMatrix(X = X2[:num_train], y = yMat)

m = 200
k = 3

epochs = 112
ir1 = .01 # irange for the first layer
mcn_out = 2.5 # max_col_norm for the output layer

predAll_train = np.zeros((num_train, 9))
predAll_test = np.zeros((test.shape[0], 9))
# 14:47:57.911620
t0 = datetime.now()
for i in range(nIter):
    seed = i + 654
    R = RImatrix(X.shape[1], m, k, rm_dup_cols = True, seed = seed)
    R = np.abs(R.todense().astype(np.float32))
    dim1 = R.shape[1]
    l1 = RectifiedLinear(layer_name='l1', irange = ir1, dim = dim1, mask_weights = R)
    l2 = RectifiedLinear(layer_name='l2', irange = ir, dim = dim, max_col_norm = 1.)
    l3 = RectifiedLinear(layer_name='l3', irange = ir, dim = dim, max_col_norm = 1.)
    l4 = RectifiedLinear(layer_name='l4', irange = ir, dim = dim, max_col_norm = 1.)
    output = Softmax(layer_name='y', n_classes = 9, irange = ir,
                      max_col_norm = mcn_out)
    mdl = MLP([l1, l2, l3, l4, output], nvis = X2.shape[1])
    trainer = sgd.SGD(learning_rate=lr,
                      batch_size=bs,
                      learning_rule=learning_rule.Momentum(mm),
                      # No dropout for the input of the first layer
                      cost=Dropout(input_include_probs =
                                   {'l2':ip,'l3':ip,'l4':ip},
                                   input_scales =
                                   {'l2':1/ip,'l3':1/ip,'l4':1/ip},
                                   default_input_include_prob=1.,
                                   default_input_scale=1.),
                      termination_criterion=EpochCounter(epochs),seed = seed)
    decay = sgd.LinearDecayOverEpoch(start=2, saturate=20, decay_factor= .1)
    #fname = path + 'model/Sparse_RI_seed_' + str(seed) + '.pkl'
    experiment = Train(dataset = training, model=mdl, algorithm=trainer, extensions=[decay])
    #                   save_path = fname, save_freq = epochs)
    experiment.main_loop()
    pred_train = predict(mdl, X_train)
    pred_test = predict(mdl, X_test)
    predAll_train += pred_train
    predAll_test += pred_test
    sc1 = log_loss(yMat, pred_train)
    sc2 = log_loss(yMat, predAll_train / (i + 1))
    print i, "  each:%f, avg:%f, Time:%s" % (sc1, sc2, datetime.now() - t0)

pred_Sparse_RI = predAll_test / nIter
np.save(path + 'pred_Sparse_RI.npy', pred_Sparse_RI)
