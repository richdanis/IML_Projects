import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

# PREPARING TRAINING FEATURES

train_df = train_df.sort_values(by=['pid','Time'])
train_df = train_df.drop(columns=['Time'])

test_df = test_df

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

num_features = [108, 24, 96, 108, 72, 36, 96, 108, 24, 120]

output = np.empty((y.shape[0],10))

for i in range(y.shape[1]):
    clf = SGDClassifier()
    X_new = SelectKBest(k=num_features[i]).fit_transform(X, y[:,i])
    clf.fit(X_new, y[:,i])
    coef = clf.coef_
    coef = coef.reshape((coef.shape[1],))
    output[:,i] = 1/(1 + np.exp(np.dot(X_new,coef)))

print(output[0,:])
print(output[output.shape[0] - 1,:])