# Imports 
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

#CSV->DataFrames
train_str = 'data_interpolated_means.csv'
labels_str = 'labels_sorted.csv'
test_str = 'test_features.csv'

df_train = pd.read_csv(train_str)
df_labels = pd.read_csv(labels_str)
df_labels = df_labels.sort_values(by=['pid'])
df_test = pd.read_csv(test_str)

#Subtasks
TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TEST2 = ['LABEL_Sepsis']
TEST3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

df_subtask1 = df_labels[TEST1]
df_subtask2 = df_labels[TEST2]
df_subtask3 = df_labels[TEST3]


# dropping the time
df_train = df_train.drop(columns=['Time'])
    	
# Normalizing the colums
df_train=(df_train-df_train.mean())/df_train.std()

# convert to numpy array
X = df_train.to_numpy()
pids = np.unique(X[:,0])
y = df_subtask1.to_numpy()

# exclude pids
X = X[:,1:]
y = y[:,1:]
age = X[::12,0]
age = age.reshape((len(age),1))

# exclude ages
X = X[:,1:]
X = X.reshape((len(pids),12*X.shape[-1]))
X = np.hstack((age,X))

#SDGClassifer
clf = SGDClassifier()

for i in range(y.shape[1]):
    print("label: " + TEST1[i])
    X_new = SelectKBest(k=120).fit_transform(X, y[:,i])
    scores = cross_val_score(clf, X_new, y[:,i], cv=5)
    print(scores)