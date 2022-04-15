import numpy as np
import pandas as pd
import sklearn.metrics as metrics

TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

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
def normalise(X):
    mean = np.mean(X,axis=0)
    mean = np.resize(mean,(1,mean.shape[0]))
    std = np.std(X,axis=0)
    std = np.resize(std,(1,std.shape[0]))
    X = (X-mean)/std
    return X

def min_mean_max(df):
    X = df.drop(columns=['pid']).to_numpy()
    age = X[::12, 0]
    age = age.reshape((len(age), 1))
    X = X[:, 1:]

    out = np.empty((X.shape[0]//12,X.shape[1]*3))
    
    for i in range(out.shape[0]):
        mean = X[(i*12):((i+1)*12)].mean(axis=0)
        min_ = X[(i*12):((i+1)*12)].min(axis=0)
        max_ = X[(i*12):((i+1)*12)].max(axis=0)
        out[i,:X.shape[1]] = mean
        out[i,X.shape[1]:2*X.shape[1]] = min_
        out[i,2*X.shape[1]:] = max_
            
    out = normalise(out)
    out = np.hstack((age, out))
    return out