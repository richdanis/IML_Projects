import pandas as pd
import numpy as np

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
    df = df.fillna(df.mean())

    X = df.to_numpy()

    # exclude pids
    X = X[:, 1:]
    ds = np.empty((X.shape[0]//12,X.shape[1]))

    for i in range(ds.shape[0]):
        ds[i] = X[(i*12):((i+1)*12)].mean(axis=0)

    return ds

def normalize(df):
    return (df - df.mean()) / df.std()