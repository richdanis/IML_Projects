import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import helper_functions as hf
from sklearn.linear_model import SGDClassifier

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")

train_df, label_df = hf.remove_outliers(train_df, label_df)

label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

columns = train_df.drop(columns=["Time","pid","Age"]).columns

X = hf.take_mean_features(train_df,["Time"])

y = label_df.to_numpy()
y = y[:,1:]

clf = SGDClassifier()
for i in range(y.shape[1]):
    print(columns[i])
    X_new = SelectKBest(k=1).fit_transform(X, y[:, i])
    scores = cross_val_score(clf, X_new, y[:,i])
    print(np.mean(scores))
'''
EtCO2
0.8182612913200028
PTT
0.931194165065766
BUN
0.7732620658057708
Lactate
0.7748965126775308
Temp
0.770865032353327
Hgb
0.8038788532150655
HCO3
0.9046088283068909
BaseExcess
0.7983219178285437
RRate
0.9678579207342006
Fibrinogen
0.9358792846718881

'''






