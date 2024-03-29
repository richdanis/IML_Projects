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

#train_df, label_df = hf.remove_sparse(train_df, label_df)
#train_df, label_df = hf.remove_outliers(train_df, label_df)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
X = hf.min_mean_max(train_df)
T = hf.min_mean_max(test_df)

# PREPARING TRAINING LABELS

y = (label_df.loc[:,"LABEL_Sepsis"]).to_numpy()

#clf = SGDClassifier()
clf = GradientBoostingClassifier()

#selector = SelectKBest(k=20)
#X_new = selector.fit_transform(X, y)
#mask = selector.get_support()
#T_new = T[:,mask]

clf.fit(X, y)
#coef = clf.coef_
#coef = coef.reshape((coef.shape[1],))
# calculate sigmoid to get probabilities in range [0,1]
#output = 1/(1 + np.exp(-np.dot(T,coef)))
output = clf.predict_proba(T)[:,1]
pids = test_df.to_numpy()[::12,0]
pids = pids.reshape((len(pids),1))
output = output.reshape((len(output),1))
output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid","LABEL_Sepsis"])
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st2.csv', index=False, float_format='%.3f', header=True)