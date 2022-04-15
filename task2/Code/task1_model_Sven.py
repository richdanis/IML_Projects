import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import helper_functions_Sven as hf

# Mit LogisticRegression
# Score in ST1: 0.7667069952488018
# Score in ST2: 0.6880016215315077

# Mit GradientBoostingClassifier
# Score in ST1: 0.8567189987308869

# Defining the columns needed in Subtask1
TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TEST2 = ['LABEL_Sepsis']
SOL  =  ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2','LABEL_Sepsis']


# Import the data
fname = 'Data/'
#df_train = pd.read_csv(fname + 'train_features_train_set.csv')
#df_label = pd.read_csv(fname + 'train_labels_train_set.csv')
#df_test = pd.read_csv(fname + 'train_features_val_set.csv')
#df_true = pd.read_csv(fname + 'train_labels_val_set.csv')

df_train = pd.read_csv(fname + 'train_features_Sven_long.csv')
df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
df_test = df_train.copy()
df_true = df_label.copy()

#df -> np matrix
X_train = hf.min_mean_max(df_train)
T_test = hf.min_mean_max(df_test)

# Labels for Subtask1_1 -> np matrix
df_subtask_1 = df_label[TEST1]
df_subtask_2 = df_label[TEST2]
y_train_1 = df_subtask_1.to_numpy()
y_train_2 = df_subtask_2.to_numpy()

# prediction matrix 
# pred_log = np.empty((T_test.shape[0],10))
pred_GBC = np.empty((T_test.shape[0],10))

for i in range(y_train_1.shape[1]):
    print(f'Round {i+1}: {TEST1[i]}')
    g = GradientBoostingClassifier()
    g.fit(X_train, y_train_1[:,i])
    pred_GBC[:,i] = g.predict_proba(T_test)[:,1]
    
    #clf = LogisticRegression(max_iter=1000)
    #clf.fit(X_train, y_train_1[:,i])
    #coef = clf.coef_
    #coef = coef.reshape((coef.shape[1],))

    # calculate sigmoid to get probabilities in range [0,1]
    #pred_log[:,i] = hf.sigmoid(T_test, coef)
    
print(f'Round 11: {TEST2}')
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train_2)
#coef = clf.coef_
#coef = coef.reshape((coef.shape[1],))
#pred_2 = hf.sigmoid(T_test,coef)
pred_2 = g.predict_proba(T_test)[:,1]

pids = df_test['pid'].to_numpy()[::12]

pids = pids.reshape((len(pids),1))
pred_2 = pred_2.reshape((len(pred_2),1))

#pred_l = np.concatenate((pids, pred_log, pred_2),axis=1)
pred_G = np.concatenate((pids, pred_GBC, pred_2),axis=1)

#df_l = pd.DataFrame(pred_l,columns=SOL)
#df_l = df_l.astype({'pid':'int32'}) 

df_G = pd.DataFrame(pred_G,columns=SOL)
df_G = df_G.astype({'pid':'int32'}) 

#score = hf.get_score(df_true, df_l,True,True)
score = hf.get_score(df_true, df_G,True,False)

#df.to_csv('Data/pred_st1_Sven.csv', index=False, float_format='%.3f',header=True)
