import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
import helper_functions as hf
from sklearn.ensemble import GradientBoostingClassifier


fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

train_df, label_df = hf.remove_outliers(train_df, label_df)

X = hf.min_mean_max(train_df)
T = hf.min_mean_max(test_df)

label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

y = label_df.to_numpy()
y = y[:,1:]

output = np.empty((T.shape[0],10))

for i in range(y.shape[1]):
    #clf = SGDClassifier()
    clf = GradientBoostingClassifier()
    # select 61 best feature columns
    # selector = SelectKBest(k=80)
    # X_new = selector.fit_transform(X, y[:,i])

    # apply mask to get the selected columns from T
    # mask = selector.get_support()
    # T_new = T[:,mask]

    clf.fit(X, y[:,i])
    #coef = clf.coef_
    #coef = coef.reshape((coef.shape[1],))

    # calculate sigmoid to get probabilities in range [0,1]
    #output[:,i] = 1/(1 + np.exp(-np.dot(T_new,coef)))
    output[:,i] = clf.predict_proba(T)[:,1]

pids = test_df.to_numpy()[::12,0]
pids = pids.reshape((len(pids),1))
output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=label_df.columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st1.csv', index=False, float_format='%.3f',header=True)

