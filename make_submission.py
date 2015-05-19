"""
Ensemble by columnwise weighted sum.
The weights are determined by scipy.optimize.minimize using validation set predictions.

LB Private: 0.40076
LB Public: 0.39773
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp

path = './'

# Neural Networks
pred = [np.load(path + 'pred_TRI_kmax_' + str(k_max) + '.npy')  for k_max in [4,5]]
pred.append(np.load(path + 'pred_Sparse_RI.npy'))
pred_NN = (pred[0] + pred[1] + pred[2]) / 3

# XGBoost
pred_XGB = (np.load(path + 'pred_RI.npy') + np.load(path + 'pred_CF.npy')) / 2

# Ensemble weights
w = np.array([1.,0.95657896,0.52392701,0.75156431,1.,0.77871818,0.81764163,0.9541003,0.82863579])

pr005 = pp.normalize(pred_NN * w + pred_XGB * (1 - w), norm = 'l1')
pred005 = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
pred005.iloc[:,:] = pr005
pred005.to_csv(path + 'pred005.csv', float_format='%.8f')
