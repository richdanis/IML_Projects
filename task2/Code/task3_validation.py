import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import helper_functions as hf
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

columns = ["pid","Time","Age","RRate","ABPm","SpO2","Heartrate"]

train_df = train_df[columns]

train_df, label_df = hf.remove_outliers(train_df, label_df)

X = hf.take_mean_features(train_df,["Time","Age"])

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

y = label_df[columns].to_numpy()
# found out these polynomial degrees through trial and error
p = [5, 7, 1, 8]
for i in range(4):
    reg = SGDRegressor()
    poly = PolynomialFeatures(p[i])
    X_new = X[:,i]
    X_new = X_new.reshape((X.shape[0],1))
    X_new = poly.fit_transform(X_new)
    X_new = hf.normalize(X_new)
    scores = cross_val_score(reg, X_new, y[:, i], cv=10)
    print("Score: ",np.mean(scores))

'''
Score:  0.1355936387732553
Score:  0.30412520112397556
Score:  0.3538507248387641
Score:  0.26148310164533284
'''