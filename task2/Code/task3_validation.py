import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import helper_functions as hf
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")

#train_df, label_df = hf.remove_outliers(train_df, label_df)

X = hf.prepare_dataset(train_df)
X = hf.normalize(X)

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

y = label_df[columns].to_numpy()
res = np.empty((4,))

for i in range(4):
    reg = SGDRegressor()
    X_new = SelectKBest(f_regression,k=61).fit_transform(X,y[:, i])
    scores = cross_val_score(reg, X_new, y[:, i])
    res[i] = 0.5 + 0.5 * np.mean(scores)

print(np.mean(res))
'''
0.7455984141492595
'''