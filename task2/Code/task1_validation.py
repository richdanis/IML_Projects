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
    X_new = SelectKBest(k=13).fit_transform(X, y[:,i])
    scores = cross_val_score(clf, X_new, y[:,i], cv=10)
    print(np.mean(scores))
    # if np.mean(scores) > max:
    #max = np.mean(scores)
    #maxIdx = j
    #print(max)
    #print("Features: ",maxIdx)

'''
label: 0
0.7792300076430765
label: 1
0.937055457183812
label: 2
0.8139607183947908
label: 3
0.8236399539783438
label: 4
0.8152471216065148
label: 5
0.8266678090482056
label: 6
0.9140904602981615
label: 7
0.7983435467682842
label: 8
0.9644486785093553
label: 9
0.9445162966166467
'''






