import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

# PREPARING TRAINING FEATURES

train_df = train_df.sort_values(by=['pid','Time'])
# train_df = train_df.drop(columns=['Time'])

# fill NaN with median for each column
train_df = train_df.fillna(train_df.mean())
train_df=(train_df-train_df.mean())/train_df.std()
# convert to numpy array
X = train_df.to_numpy()
pids = np.unique(X[:,0])

# exclude pids
X = X[:,1:]
age = X[::12,0]
age = age.reshape((len(age),1))

# exclude ages
X = X[:,1:]
X = X.reshape((len(pids),12*X.shape[-1]))
X = np.hstack((age,X))

# PREPARING TRAINING LABELS
label_df = label_df.sort_values(by=['pid'])
label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

y = label_df.to_numpy()
y = y[:,1:]

clf = SGDClassifier()
for i in range(y.shape[1]):
    print("label: " + str(i))
    max = 0
    maxIdx = 12
    for j in range(13,132,12):
        X_new = SelectKBest(k=j).fit_transform(X, y[:,i])
        scores = cross_val_score(clf, X_new, y[:,i], cv=10)
        if np.mean(scores) > max:
            max = np.mean(scores)
            maxIdx = j
    print(max)
    print("Features: ",maxIdx)

#for i in range(y.shape[1]):
#    print("label: " + str(i))
#    X_new = SelectKBest(k=12).fit_transform(X, y[:,i])
#    scores = cross_val_score(clf, X_new, y[:,i], cv=10)
#    print(np.mean(scores))

'''
label: 0
0.7725716859288823
label: 1
0.9227684653973005
label: 2
0.7665174745711039
label: 3
0.770307724287021
label: 4
0.7648852858845376
label: 5
0.8095817466256479
label: 6
0.9018157201851391
label: 7
0.7889439039937918
label: 8
0.9603579723400127
label: 9
0.9279820958399159
'''






