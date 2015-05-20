# kaggle-Otto
# 13th solution for the Otto Group Product Classification Challenge on Kaggle.  

## Classifier algorithms 
* NN3: Neural Networks with 3 hidden layers  
* NN4: Neural Networks with 4 hidden layers. The first is sparse. The link is defined by RI or TRI.
* XGB: XGBoost

## Feature extraction methods  
* NewX = X * R,  X:(N, 93), R:(93, m)  
    * Random Indexing(RI): RI controls # of nonzero elements in each row of R.  
    * Transposed Random Indexing(TRI): TRI controls # of nonzero elements in each column of R.  
* Count Feature(CF): The count of each feature value  

## 5 types of models   
* TRI(k_max=4) + NN3
* TRI(k_max=5) + NN3
* NN4(RI)
* XGB + CF
* XGB + RI

## Software
* python 2.7
* numpy
* scipy
* scikit-learn 
* theano 0.7
* pylearn2

## Usage
* Put all *.py files into the folder containing train.csv, test.csv, sampleSubmission.csv
* python otto_nn.py
    * Output files: pred_TRI_kmax_4.npy, pred_TRI_kmax_5.npy, pred_Sparse_RI.npy
* python otto_xgb.py
    * Output files: pred_XGB_CF.npy, pred_XGB_RI.npy
* python make_submission.py
    * Output file: pred005.csv

More information comming soon!
