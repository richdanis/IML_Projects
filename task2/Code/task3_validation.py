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

train_df, label_df = hf.remove_outliers(train_df, label_df)

X = hf.prepare_dataset(train_df,["Time"])

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

y = label_df[columns].to_numpy()
poly = PolynomialFeatures(2)

for i in range(4):
    reg = SGDRegressor()
    X_new = SelectKBest(f_regression, k=13).fit_transform(X, y[:, i])
    X_new = poly.fit_transform(X_new)
    scores = cross_val_score(reg, X_new, y[:, i], cv=10)
    print(np.mean(scores))

'''
0.3201414280639009
0.45985973613410325
0.3768117427990279
0.597573914871371
'''