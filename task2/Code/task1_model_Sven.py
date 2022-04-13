import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

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

def prepare_dataset(df, drop):

    if not drop == None:
        df = df.drop(columns=drop)
    df = (df - df.mean()) / df.std()

    X = df.to_numpy()

    # exclude pids
    X = X[:, 1:]
    # exclude ages
    X = X[:, 1:]
    X = X.reshape((len(age), 12 * X.shape[-1]))
    X = np.hstack((age, X))

    return X

fname = 'Data/'
train_df = pd.read_csv(fname + 'data_interpolated_means.csv')
label_df = pd.read_csv(fname + 'train_labels.csv')
test_df = pd.read_csv(fname + 'testdata_means_long.csv')

X_train = prepare_dataset(train_df, 'Time')
T_test = prepare_dataset(test_df, None)

# PREPARING TRAINING LABELS
label_df = label_df.sort_values(by=['pid'])
label_df = label_df[TEST1]

y = label_df.to_numpy()
y = y[:,1:]

output = np.empty((T.shape[0],10))

for i in range(y.shape[1]):
    clf = SGDClassifier()

    # select 61 best feature columns
    selector = SelectKBest(k=61)
    X_new = selector.fit_transform(X_train, y[:,i])

    # apply mask to get the selected columns from T
    mask = selector.get_support()
    T_new = T[:,mask]

    clf.fit(X_new, y[:,i])
    coef = clf.coef_
    coef = coef.reshape((coef.shape[1],))

    # calculate sigmoid to get probabilities in range [0,1]
    output[:,i] = 1/(1 + np.exp(-np.dot(T_new,coef)))

pids = test_df.to_numpy()[::12,0]
pids = pids.reshape((len(pids),1))
output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=label_df.columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st1_Sven.csv', index=False, float_format='%.3f',header=True)

