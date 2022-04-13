import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import helper_functions as hf
from sklearn.preprocessing import PolynomialFeatures

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

columns = ["pid","Time","Age","RRate","ABPm","SpO2","Heartrate"]

train_df = train_df[columns]
test_df = test_df[columns]

train_df, label_df = hf.remove_outliers(train_df,label_df)

X = hf.take_mean_features(train_df,["Time","Age"])
T = hf.take_mean_features(test_df,["Time","Age"])

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

pids = test_df["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

y = label_df[columns].to_numpy()

output = np.empty((pids.shape[0],4))

p = [5, 7, 1, 8]

for i in range(4):
    reg = SGDRegressor()
    poly = PolynomialFeatures(p[i])
    X_new = X[:,i]
    T_new = T[:,i]
    X_new = X_new.reshape((X_new.shape[0],1))
    T_new = T_new.reshape((T_new.shape[0], 1))
    X_new = poly.fit_transform(X_new)
    T_new = poly.fit_transform(T_new)
    X_new = hf.normalize(X_new)
    T_new = hf.normalize(T_new)
    reg.fit(X_new,y[:,i])
    output[:,i] = reg.predict(T_new)

output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid"] + columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st3.csv', index=False, float_format='%.3f', header=True)