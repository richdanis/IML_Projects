import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest, f_regression

def prepare_dataset(df, columns):

    df = df.drop(columns=columns)
    df = df.fillna(df.mean())
    df = (df - df.mean()) / df.std()

    X = df.to_numpy()

    # exclude pids
    X = X[:, 1:]
    age = X[::12, 0]
    age = age.reshape((len(age), 1))

    # exclude ages
    X = X[:, 1:]
    X = X.reshape((len(age), 12 * X.shape[-1]))
    X = np.hstack((age, X))

    return X

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

X = prepare_dataset(train_df,["Time"])
T = prepare_dataset(test_df,["Time"])

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

pids = test_df["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

y = label_df[columns].to_numpy()

output = np.empty((pids.shape[0],4))

for i in range(4):
    reg = SGDRegressor()
    selector = SelectKBest(f_regression,k=61)
    X_new = selector.fit_transform(X, y[:,i])
    mask = selector.get_support()
    T_new = T[:, mask]

    reg.fit(X_new,y[:,i])
    output[:,i] = reg.predict(T_new)

output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid"] + columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st3.csv', index=False, float_format='%.3f', header=True)