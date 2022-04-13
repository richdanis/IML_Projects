import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import helper_functions as hf
from sklearn.preprocessing import PolynomialFeatures

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

train_df, label_df = hf.remove_outliers(train_df,label_df)

X = hf.prepare_dataset(train_df,["Time"])
T = hf.prepare_dataset(test_df,["Time"])

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

pids = test_df["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

y = label_df[columns].to_numpy()
# poly = PolynomialFeatures(2)

output = np.empty((pids.shape[0],4))

for i in range(4):
    reg = SGDRegressor()
    selector = SelectKBest(f_regression,k=13)
    X_new = selector.fit_transform(X, y[:,i])
    # X_new = poly.fit_transform(X_new)
    mask = selector.get_support()
    T_new = T[:, mask]
    # T_new = poly.fit_transform(T_new)

    reg.fit(X_new,y[:,i])
    output[:,i] = reg.predict(T_new)

output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid"] + columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st3.csv', index=False, float_format='%.3f', header=True)