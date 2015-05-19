# kaggle-Otto
# 13th solution for the Otto Group Product Classification Challenge on Kaggle.  

## Classifier algorithms 
* NN3: Neural Networks with 3 hidden layers  
* NN4: Neural Networks with 4 hidden layers. The first is sparse. The link is defined by RI or TRI.
* XGB: XGBoost

## Feature extraction  
* NewX = X * R,  X:(N, 93), R:(93, m)  
    * Random Indexing(RI): RI controls # of nonzero elements in each row of R.  
    * Transposed Random Indexing(TRI): TRI controls # of nonzero elements in each column of R.  
* Count Feature(CF): The count of each feature value  

## 5 types of models.  
* TRI(k_max=4) + NN3  
* TRI(k_max=5) + NN3  
* NN4(RI)  
* CF + XGB  
* RI + XGB

More information comming soon!
