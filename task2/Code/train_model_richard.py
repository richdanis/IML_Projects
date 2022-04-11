import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

# PREPARING TRAINING FEATURES

train_df = train_df.sort_values(by=['pid','Time'])
train_df = train_df.drop(columns=['Time'])

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

#poly = PolynomialFeatures(10)

# PREPARING TRAINING LABELS
label_df = label_df.sort_values(by=['pid'])
label_df = label_df.drop(columns=['LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

y = label_df.to_numpy()
y = y[:,1:]

#accuracy = np.empty((y.shape[1],))

clf = SGDClassifier()
for i in range(y.shape[1]):
    print("label: " + str(i))
    X_new = SelectKBest(k=1).fit_transform(X, y[:,i])
    #X_new = poly.fit_transform(X_new)
    scores = cross_val_score(clf, X_new, y[:,i], cv=10)
    print(scores)

#for i in range(y.shape[1]):
    #clf = LogisticRegressionCV(cv=10, max_iter=100, solver='sag').fit(X,y[:,i])
    #accuracy[i] = clf.score(X,y)









