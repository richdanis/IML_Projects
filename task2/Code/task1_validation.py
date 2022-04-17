import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import helper_functions as hf
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")

label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

#train_df, label_df = hf.remove_sparse(train_df, label_df)
#train_df, label_df = hf.remove_outliers(train_df, label_df)

train_df = train_df.fillna(0)
X = hf.min_mean_max(train_df)

y = label_df.to_numpy()
y = y[:,1:]

#clf = SGDClassifier()
clf = GradientBoostingClassifier()

means = np.empty((y.shape[1],))

for i in range(y.shape[1]):
    #X_new = SelectKBest(k=80).fit_transform(X, y[:, i])
    scores = cross_val_score(clf, X, y[:,i],scoring='roc_auc')
    means[i] = np.mean(scores)

print(np.mean(means))
'''
0.8015024040964818>
'''






