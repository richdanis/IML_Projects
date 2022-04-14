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

X = hf.prepare_dataset(train_df,["Time"])
X = hf.normalize(X)

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

y = label_df[columns].to_numpy()

for i in range(4):
    reg = SGDRegressor()
    X_new = SelectKBest(f_regression,k=50).fit_transform(X,y[:, i])
    scores = cross_val_score(reg, X_new, y[:, i])
    print("R2: ",np.mean(scores))

'''
Score:  0.37890204825513146
Score:  0.5710904208631888
Score:  0.32837099160767363
Score:  0.5988897768359733
'''