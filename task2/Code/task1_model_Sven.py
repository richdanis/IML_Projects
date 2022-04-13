import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

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

def prepare_dataset(df):
    df = df.drop('pids')
    df = (df - df.mean()) / df.std()

    X = df.to_numpy()
    return X

fname = 'Data/'
df_train = pd.read_csv(fname + 'train_features_Sven_short.csv')
df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
df_test = pd.read_csv(fname + 'test_features_Sven_short.csv')

X_train = prepare_dataset(df_train)
T_test = prepare_dataset(df_test)

# PREPARING TRAINING LABELS
df_label = df_label.sort_values(by=['pid'])
df_label = df_label[TEST1]

y_train = df_label.to_numpy()

output = np.empty((T_test.shape[0],10))

for i in range(y_train.shape[1]):
    clf = SGDClassifier()

    # select 61 best feature columns
    selector = SelectKBest(k=61)
    X_new = selector.fit_transform(X_train, y_train[:,i])

    # apply mask to get the selected columns from T
    mask = selector.get_support()
    T_new = T_test[:,mask]

    clf.fit(X_new, y_train[:,i])
    coef = clf.coef_
    coef = coef.reshape((coef.shape[1],))

    # calculate sigmoid to get probabilities in range [0,1]
    output[:,i] = 1/(1 + np.exp(-np.dot(T_new,coef)))

pids = df_test.to_numpy()[::12,0]
pids = pids.reshape((len(pids),1))
output = np.hstack((pids, output))
df = pd.DataFrame(output,columns=df_label.columns)
df = df.astype({'pid':'int32'})
df.to_csv('Data/pred_st1_Sven.csv', index=False, float_format='%.3f',header=True)

