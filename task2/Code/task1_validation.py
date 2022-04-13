import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import helper_functions as hf

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")

train_df, label_df = hf.remove_outliers(train_df, label_df)

label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

X = hf.prepare_dataset(train_df,["Time"])

y = label_df.to_numpy()
y = y[:,1:]

clf = SGDClassifier()
for i in range(y.shape[1]):
    print("label: " + str(i))
    #max = 0
    #maxIdx = 12
    #for j in range(13,132,12):
    X_new = SelectKBest(k=61).fit_transform(X, y[:,i])
    scores = cross_val_score(clf, X_new, y[:,i], cv=10)
    print(scores)
    # if np.mean(scores) > max:
    #max = np.mean(scores)
    #maxIdx = j
    #print(max)
    #print("Features: ",maxIdx)

'''
label: 0
0.779047725949946
Features:  121
label: 1
0.9264010975305563
Features:  61
label: 2
0.7689915190820653
Features:  85
label: 3
0.7738351209777999
Features:  85
label: 4
0.7697289432111084
Features:  73
label: 5
0.812582716665281
Features:  109
label: 6
0.9026582411795683
Features:  73
label: 7
0.7897859538261134
Features:  61
label: 8
0.9624638175216873
Features:  13
label: 9
0.931140378592611
Features:  25
'''






