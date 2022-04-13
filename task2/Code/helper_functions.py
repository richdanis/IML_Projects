import pandas as pd
import numpy as np

def remove_outliers(features,labels):
    for col in features.columns:
        if col == 'pid' or col == 'Age' or col == 'Time':
            continue
        # RRate filter size 3000!
        # ABPm 2761
        # ABPd 1699
        # Heartrate 2011
        # ABPs 1699
        q_low = features[col].quantile(0.01)
        q_hi = features[col].quantile(0.99)

        filter = features[(features[col] > q_hi) | (features[col] < q_low)]
        if filter.shape[0] > 500:
            continue
        features = features[~features['pid'].isin(filter.pid)]
        labels = labels[~labels['pid'].isin(filter.pid)]

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