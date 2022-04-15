import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import helper_functions_Sven as hf

# Mit LogisticRegression
# poly grad 2
# Score in ST2: 0.8843281176601985

# Mit GradientBoostingClassifier
# Score in ST2: 0.48342762668394545

# Defining the columns needed in Subtask2
TEST2 = ['LABEL_Sepsis']
SOL  =  ['pid', 'LABEL_Sepsis']

# Import the data
fname = 'Data/'
df_train = pd.read_csv(fname + 'train_features_Sven_long.csv')
df_label = pd.read_csv(fname + 'train_labels_sorted.csv')
df_test = df_train.copy()
#df_test = pd.read_csv(fname + 'test_features_Sven_long.csv')
df_true = df_label.copy()

#df -> np matrix
X_train = hf.min_mean_max(df_train)
T_test = hf.min_mean_max(df_test)

poly = PolynomialFeatures(2)
X_train = poly.fit_transform(X_train)
T_test = poly.fit_transform(T_test)

# Labels for Subtask1_1 -> np matrix
df_subtask_2 = df_label[TEST2]
y_train_2 = df_subtask_2.to_numpy()
y_train_2 = np.ravel(y_train_2)

# prediction matrix 
pred_2 = np.empty((T_test.shape[0],1))

print(f'Subtask2: {TEST2[0]}')
#clf = LogisticRegression(penalty='l1',max_iter=1000,solver='liblinear')
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train_2)
#coef = clf.coef_
#coef = coef.reshape((coef.shape[1],))
pred_2 = clf.predict_proba(T_test)[:,1]
#pred_2 = hf.sigmoid(T_test,coef)
print(pred_2.shape)
pids = df_test['pid'].to_numpy()[::12]

pred = np.empty((T_test.shape[0],2))
pred[:,0] = pids
pred[:,1] = pred_2 

df = pd.DataFrame(pred,columns=SOL)
df = df.astype({'pid':'int32'}) 

score = hf.get_score(df_true, df, False, True)

#df.to_csv('Data/pred_st2_Sven.csv', index=False, float_format='%.3f',header=True)
