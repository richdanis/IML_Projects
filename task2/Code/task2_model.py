import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

def prepare_dataset(df, columns):

    df = df.drop(columns=columns)
    df = df.fillna(df.mean())
    df = (df - df.mean()) / df.std()

    X = df.to_numpy()

    # exclude pids
    X = X[:, 1:]
    age = X[::12, 0]
    age = age.reshape((len(age), 1))

    # exclude ages
    X = X[:, 1:]
    X = X.reshape((len(age), 12 * X.shape[-1]))
    X = np.hstack((age, X))

    return X

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

X = prepare_dataset(train_df,["Time"])
T = prepare_dataset(test_df,["Time"])

# PREPARING TRAINING LABELS

y = (label_df.loc[:,"LABEL_Sepsis"]).to_numpy()

clf = SGDClassifier()

selector = SelectKBest(k=61)
X_new = selector.fit_transform(X, y)
mask = selector.get_support()
T_new = T[:,mask]

clf.fit(X_new, y)
coef = clf.coef_
coef = coef.reshape((coef.shape[1],))
# calculate sigmoid to get probabilities in range [0,1]
output = 1/(1 + np.exp(-np.dot(T_new,coef)))


pids = test_df.to_numpy()[::12,0]
pids = pids.reshape((len(pids),1))
output = output.reshape((len(output),1))
output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=["pid","LABEL_Sepsis"])
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st2.csv', index=False, float_format='%.3f', header=True)

