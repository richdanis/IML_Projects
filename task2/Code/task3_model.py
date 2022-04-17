import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import helper_functions as hf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

#train_df, label_df = hf.remove_sparse(train_df, label_df)
#train_df, label_df = hf.remove_outliers(train_df, label_df)

train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())
X = hf.min_mean_max(train_df)
T = hf.min_mean_max(test_df)
X = hf.normalize(X)
T = hf.normalize(T)

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

pids = test_df["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

y = label_df[columns].to_numpy()

output = np.empty((pids.shape[0],4))

for i in range(4):
    #reg = SGDRegressor()
    #selector = SelectKBest(f_regression, k=61)
    #X_new = selector.fit_transform(X, y[:, i])
    #mask = selector.get_support()
    #T_new = T[:, mask]
    reg = GradientBoostingRegressor()
    reg.fit(X,y[:,i])
    output[:,i] = reg.predict(T)

output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid"] + columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st3.csv', index=False, float_format='%.3f', header=True)