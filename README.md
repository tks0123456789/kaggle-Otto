# kaggle-Otto
14th solution for the Otto Group Product Classification Challenge on Kaggle.

Model
NN3: Neural Networks with 3 hidden layers
NN4: Neural Networks with 4 hidden layers, the first is sparse. The link is defined by RI or TRI.
XGB: XGBoost

Feature extraction
Count Feature(CF): The count of each feature value
NewX = X * R, X:(N, 93), R:(93, m)
Random Indexing(RI):RI controls # of nonzero elements in each row.
Transposed Random Indexing(TRI):TRI controls # of nonzero elements in each column.

I built 5 types of models.
TRI + NN3(k_max=4)
TRI + NN3(k_max=5)
NN4(RI)
CF + XGB
RI + XGB
More information comming soon!
