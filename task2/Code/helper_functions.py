import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest

def remove_sparse(features,labels):
    filter = features.groupby(['pid'], as_index=False).count()
    filter['sum'] = filter[filter.columns[1:]].sum(axis=1)
    filter = filter[filter['sum'] < 70]
    features = features[~features['pid'].isin(filter.pid)]
    labels = labels[~labels['pid'].isin(filter.pid)]
    return features, labels

def remove_outliers(features,labels):
    for col in features.columns:
        if col == 'pid' or col == 'Time':
            continue

        if col == 'Age':
            q_high = 90
            upper = features[features[col] > q_high]
            features = features[~features['pid'].isin(upper.pid)]
            labels = labels[~labels['pid'].isin(upper.pid)]
            continue

        if col == 'Calcium':
            q_low = 2.5
            lower = features[features[col] < q_low]
            features = features[~features['pid'].isin(lower.pid)]
            labels = labels[~labels['pid'].isin(lower.pid)]
            continue

        q_low = features[col].quantile(0.01)
        q_hi = features[col].quantile(0.99)

        upper = features[features[col] > q_hi].groupby(['pid'],as_index=False).max().sort_values(by=[col],ascending=False)
        lower = features[features[col] < q_low].groupby(['pid'],as_index=False).min().sort_values(by=[col])

        upper = upper.head(20)
        lower = lower.head(20)

        features = features[~features['pid'].isin(upper.pid)]
        features = features[~features['pid'].isin(lower.pid)]
        labels = labels[~labels['pid'].isin(upper.pid)]
        labels = labels[~labels['pid'].isin(lower.pid)]

    return features, labels

def prepare_dataset(df):

    df = df.fillna(df.mean())

    X = df.to_numpy()

    age = X[::12, 0]
    age = age.reshape((len(age), 1))

    # exclude pids, age and time
    X = X[:, 3:]

    # in community , sh ip bgp community
    # out filter outgoing
    # eigene hat keine community value!

    X = X.reshape((len(age), 12 * X.shape[-1]))
    X = np.hstack((age, X))

    return X

def take_mean_features(df):

    df = df.fillna(df.mean())

    X = df.to_numpy()

    # iterative imputer
    #imp_mean = IterativeImputer(random_state=0)
    #X = imp_mean.fit_transform(X)
    # show ip bgp regex

    ds = np.empty((X.shape[0] // 12, X.shape[1]))
    for i in range(ds.shape[0]):
        mean = X[(i*12):((i+1)*12)].mean(axis=0)
        ds[i] = mean

    # exclude pids and time
    X = X[:, 2:]
    return ds

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    mean = np.resize(mean, (1, mean.shape[0]))
    std = np.resize(std, (1, std.shape[0]))
    return (X - mean) / std

def min_mean_max(df):

    #df = df.fillna(df.mean())
    #df = df.fillna(0)

    X = df.to_numpy()
    # exclude pids age and time
    age = X[::12, 0]
    age = age.reshape((len(age), 1))
    X = X[:, 3:]

    # iterative imputer
    #imp_mean = IterativeImputer(random_state=0)
    #X = imp_mean.fit_transform(X)

    ds = np.empty((X.shape[0]//12,X.shape[1]*4))

    for i in range(ds.shape[0]):
        mean = X[(i*12):((i+1)*12)].mean(axis=0)
        min = X[(i*12):((i+1)*12)].min(axis=0)
        max = X[(i*12):((i+1)*12)].max(axis=0)
        last = X[(i+1) * 12 - 1]
        ds[i,:X.shape[1]] = mean
        ds[i,X.shape[1]:2*X.shape[1]] = min
        ds[i,2*X.shape[1]:3*X.shape[1]] = max
        ds[i,3*X.shape[1]:] = last

    ds = np.hstack((age,ds))

    return ds