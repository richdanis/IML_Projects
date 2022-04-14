import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest


# Defining the columns needed in Subtask1
TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TEST2 = ['LABEL_Sepsis']
SOL  =  ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2','LABEL_Sepsis']



#get score
def get_score(df_true, df_submission, T1 = False, T2 = False):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    if T1:
        task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TEST1])
        print(f'Score in ST1: {task1}')
    if T2:
        task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
        print(f'Score in ST2: {task2}')
    return

# Sigmoind
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
df_train = pd.read_csv(fname + 'train_features_train_set.csv')
df_label = pd.read_csv(fname + 'train_labels_train_set.csv')
df_test = pd.read_csv(fname + 'train_features_val_set.csv')
df_true = pd.read_csv(fname + 'train_labels_val_set.csv')

#df_train = pd.read_csv(fname + 'train_features_Sven_short.csv')
#df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
#df_test = df_train.copy()
#df_true = df_label.copy()

#df -> np matrix
X_train = prepare_dataset(df_train)
T_test = prepare_dataset(df_test)

# Labels for Subtask1_1 -> np matrix
df_subtask_1 = df_label[TEST1]
df_subtask_2 = df_label[TEST2]
y_train_1 = df_subtask_1.to_numpy()
y_train_2 = df_subtask_2.to_numpy()

# prediction matrix 
pred_1 = np.empty((T_test.shape[0],10))

for i in range(y_train_1.shape[1]):
    clf = LogisticRegression()
    clf.fit(X_train, y_train_1[:,i])
    coef = clf.coef_
    coef = coef.reshape((coef.shape[1],))

    # calculate sigmoid to get probabilities in range [0,1]
    pred_1[:,i] = sigmoid(T_test, coef)
    

clf = LogisticRegression()
clf.fit(X_train, y_train_2)
coef = clf.coef_
coef = coef.reshape((coef.shape[1],))
pred_2 = sigmoid(T_test,coef)

pids = df_test['pid'].to_numpy()

pids = pids.reshape((len(pids),1))
pred_2 = pred_2.reshape((len(pred_2),1))
pred = np.concatenate((pids, pred_1, pred_2),axis=1)

df = pd.DataFrame(pred,columns=SOL)
df = df.astype({'pid':'int32'}) 

score = get_score(df_true, df,True,True)
#df.to_csv('Data/pred_st1_Sven.csv', index=False, float_format='%.3f',header=True)


# # select 61 best feature columns
# selector = SelectKBest(k=61)
# X_new = selector.fit_transform(X_train, y_train[:,i])

# apply mask to get the selected columns from T
# mask = selector.get_support()
# T_new = T_test[:,mask]