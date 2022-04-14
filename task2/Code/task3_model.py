import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import helper_functions as hf
from sklearn.feature_selection import SelectKBest, f_regression

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

X = hf.prepare_dataset(train_df)
T = hf.prepare_dataset(test_df)
X = hf.normalize(X)
T = hf.normalize(T)

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

pids = test_df["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

y = label_df[columns].to_numpy()

output = np.empty((pids.shape[0],4))

for i in range(4):
    reg = SGDRegressor()
    selector = SelectKBest(f_regression, k=61)
    X_new = selector.fit_transform(X, y[:, i])
    mask = selector.get_support()
    T_new = T[:, mask]
    reg.fit(X_new,y[:,i])
    output[:,i] = reg.predict(T_new)

output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid"] + columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st3.csv', index=False, float_format='%.3f', header=True)