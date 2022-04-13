import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import helper_functions as hf
from sklearn.svm import LinearSVC

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")

train_df, label_df = hf.remove_outliers(train_df, label_df)

label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

X = hf.prepare_dataset(train_df,["Time"])

y = label_df.to_numpy()
y = y[:,1:]

clf = LinearSVC(dual=False)
for i in range(y.shape[1]):
    print("label: " + str(i))
    X_new = SelectKBest(k=13).fit_transform(X, y[:,i])
    scores = cross_val_score(clf, X_new, y[:,i])
    print(np.mean(scores))
'''
label: 0
0.7829673257626789
label: 1
0.9460116060290137
label: 2
0.7924821562200108
label: 3
0.7958564308431975
label: 4
0.7923471806804845
label: 5
0.8318260115878106
label: 6
0.9035631993871333
label: 7
0.8076664831163296
label: 8
0.9742206284162339
label: 9
0.9441894476319046
'''






