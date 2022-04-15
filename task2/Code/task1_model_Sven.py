import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import helper_functions_Sven as hf

# Mit LogisticRegression
# Score in ST1: 0.7667069952488018

# Mit GradientBoostingClassifier
# Score in ST1: 0.8567189987308869

# Defining the columns needed in Subtask1
TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
SOL  =  ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

# Import the data
fname = 'Data/'
df_train = pd.read_csv(fname + 'train_features_Sven_long.csv')
df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
# df_test = df_train.copy()
df_test = pd.read_csv(fname + 'test_features_Sven_long.csv')
df_true = df_label.copy()

#df -> np matrix
X_train = hf.min_mean_max(df_train)
T_test = hf.min_mean_max(df_test)

# Labels for Subtask1_1 -> np matrix
df_subtask_1 = df_label[TEST1]
y_train_1 = df_subtask_1.to_numpy()

# prediction matrix 
pred_GBC = np.empty((T_test.shape[0],10))

print(f'Subtask1:')
for i in range(y_train_1.shape[1]):
    print(f'Round {i+1}: {TEST1[i]}')
    g = GradientBoostingClassifier()
    g.fit(X_train, y_train_1[:,i])
    pred_GBC[:,i] = g.predict_proba(T_test)[:,1]
    
pids = df_test['pid'].to_numpy()[::12]

pids = pids.reshape((len(pids),1))
pred = np.hstack((pids, pred_GBC))

df = pd.DataFrame(pred,columns=SOL)
df = df.astype({'pid':'int32'}) 

#score = hf.get_score(df_true, df, True, False)

df.to_csv('Data/pred_st1_Sven.csv', index=False, float_format='%.3f',header=True)
