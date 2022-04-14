import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def remove_outliers(features,labels):
    for col in features.columns:
        if col == 'pid' or col == 'Age' or col == 'Time':
            continue

        q_low = features[col].quantile(0.01)
        q_hi = features[col].quantile(0.99)

        upper = features[features[col] > q_hi].groupby(['pid'],as_index=False).max().sort_values(by=[col],ascending=False)
        lower = features[features[col] < q_low].groupby(['pid'],as_index=False).min().sort_values(by=[col])

        upper = upper.head(10)
        lower = lower.head(10)

        features = features[~features['pid'].isin(upper.pid)]
        features = features[~features['pid'].isin(lower.pid)]
        labels = labels[~labels['pid'].isin(upper.pid)]
        labels = labels[~labels['pid'].isin(lower.pid)]

    return features, labels

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

def take_mean_features(df, columns):

    df = df.drop(columns=columns)
    #df = df.fillna(df.mean())

    X = df.to_numpy()
    # exclude pids
    X = X[:, 1:]

    # iterative imputer
    imp_mean = IterativeImputer(random_state=0)
    X = imp_mean.fit_transform(X)

    ds = np.empty((X.shape[0]//12,X.shape[1]*3))

    for i in range(ds.shape[0]):
        mean = X[(i*12):((i+1)*12)].mean(axis=0)
        min = X[(i*12):((i+1)*12)].min(axis=0)
        max = X[(i*12):((i+1)*12)].max(axis=0)
        ds[i,:X.shape[1]] = mean
        ds[i,X.shape[1]:2*X.shape[1]] = min
        ds[i,2*X.shape[1]:] = max

    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    mean = np.resize(mean,(1,mean.shape[0]*3))
    std = np.resize(std,(1,std.shape[0]*3))
    return (ds - mean) / std

def normalize(df):
    return (df - df.mean()) / df.std()
