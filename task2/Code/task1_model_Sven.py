import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest


# Defining the columns needed in Subtask1
TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

SOL1= ['pid','LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def sigmoid(T, coef):
    sol = 1/(1 + np.exp(-np.dot(T,coef)))
    return sol 
    
# Normalizing and retun a np matrix
def prepare_dataset(df):
    df = df.drop(columns='pid')
    X = df.to_numpy()
    mean = np.mean(X,axis=0)
    mean = np.resize(mean,(1,mean.shape[0]))
    std = np.std(X,axis=0)
    std = np.resize(std,(1,std.shape[0]))
    X = (X-mean)/std
    return X

# Import the data
fname = 'Data/'
df_train = pd.read_csv(fname + 'train_features_Sven_short.csv')
df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
df_test = pd.read_csv(fname + 'test_features_Sven_short.csv')

#df -> np matrix
X_train = prepare_dataset(df_train)
T_test = prepare_dataset(df_test)

# Labels for Subtask1_1 -> np matrix
df_subtaskt_1 = df_label[TEST1]
y_train = df_subtaskt_1.to_numpy()

# prediction matrix 
output = np.empty((T_test.shape[0],10))

for i in range(y_train.shape[1]):
    clf = SGDClassifier(loss='log')

    # # select 61 best feature columns
    # selector = SelectKBest(k=61)
    # X_new = selector.fit_transform(X_train, y_train[:,i])

    # apply mask to get the selected columns from T
    # mask = selector.get_support()
    # T_new = T_test[:,mask]

    clf.fit(X_train, y_train[:,i])
    coef = clf.coef_
    coef = coef.reshape((coef.shape[1],))

    # calculate sigmoid to get probabilities in range [0,1]
    output[:,i] = sigmoid(T_test, coef)


pids = df_test['pid'].to_numpy()
pids = pids.reshape((len(pids),1))

output = np.hstack((pids, output))
df_sig = pd.DataFrame(output,columns=SOL1)
df_sig = df_sig.astype({'pid':'int32'})

df_sig.to_csv('Data/pred_st1_Sven.csv', index=False, float_format='%.3f',header=True)

