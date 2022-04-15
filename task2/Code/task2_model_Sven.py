import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import helper_functions_Sven as hf

# Mit LogisticRegression
# Score in ST2: 0.6879623561604237

# Mit GradientBoostingClassifier
# Score in ST2: 0.8137293397422631

# Defining the columns needed in Subtask2
TEST2 = ['LABEL_Sepsis']
SOL  =  ['pid', 'LABEL_Sepsis']

# Import the data
fname = 'Data/'
#df_train = pd.read_csv(fname + 'train_features_train_set.csv')
#df_label = pd.read_csv(fname + 'train_labels_train_set.csv')
#df_test = pd.read_csv(fname + 'train_features_val_set.csv')
#df_true = pd.read_csv(fname + 'train_labels_val_set.csv')

df_train = pd.read_csv(fname + 'train_features_Sven_long.csv')
df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
#df_test = df_train.copy()
df_test = pd.read_csv(fname + 'test_features_Sven_long.csv')
df_true = df_label.copy()

#df -> np matrix
#X = df_train.to_numpy()
#T = df_test.copy()
X_train = hf.min_mean_max(df_train)
T_test = hf.min_mean_max(df_test)

#poly = PolynomialFeatures(2)
#X_train = poly.fit_transform(X_train)
#T_test = poly.fit_transform(T_test)

# Labels for Subtask1_1 -> np matrix
df_subtask_2 = df_label[TEST2]
y_train_2 = df_subtask_2.to_numpy()
y_train_2 = np.ravel(y_train_2)

# prediction matrix 
pred_2 = np.empty((T_test.shape[0],1))
pred_GBC = np.empty((T_test.shape[0],1))

print(f'Subtask2: {TEST2[0]}')
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train_2)
#pred_2 = clf.predict_proba(T_test)[:,1]


g = GradientBoostingClassifier()
g.fit(X_train, y_train_2)
pred_GBC = g.predict_proba(T_test)[:,1]

pids = df_test['pid'].to_numpy()[::12]
#pids = df_test['pid'].to_numpy()

#pred = np.empty((T_test.shape[0],2))
pred_G = np.empty((T_test.shape[0],2))
#pred[:,0] = pids
#pred[:,1] = pred_2 
pred_G[:,0] = pids
pred_G[:,1] = pred_GBC 

#df = pd.DataFrame(pred,columns=SOL)
#df = df.astype({'pid':'int32'}) 

df2 = pd.DataFrame(pred_G,columns=SOL)
df2 = df2.astype({'pid':'int32'}) 

#score = hf.get_score(df_true, df, False, True)
#score = hf.get_score(df_true, df2, False, True)

df2.to_csv('Data/pred_st2_Sven.csv', index=False, float_format='%.3f',header=True)
