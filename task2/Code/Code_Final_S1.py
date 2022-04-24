#imports
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

# label for the Subtasks 1-3
TASK1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TASK2 = ['LABEL_Sepsis']
TASK3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

# drop pid, time and age column
# then for each patient, take the min, mean, max and last of all feature columns
# results in 34 * 4 = 136 features per patient
# add age back, 137 features per patient

def pre_pros(df):

    df = df.drop(columns=['pid'])
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    X = df.to_numpy()
    
    age = X[::12, 0]
    age = age.reshape((len(age), 1))
    X = X[:, 1:]

    ds = np.empty((X.shape[0]//12,X.shape[1]*4))

    for i in range(ds.shape[0]):
        mean = X[(i*12):((i+1)*12)].mean(axis=0)
        min = X[(i*12):((i+1)*12)].min(axis=0)
        max = X[(i*12):((i+1)*12)].max(axis=0)
        last = X[(i+1) * 12 - 1]
        ds[i,:X.shape[1]] = mean
        ds[i,1*X.shape[1]:2*X.shape[1]] = min
        ds[i,2*X.shape[1]:3*X.shape[1]] = max
        ds[i,3*X.shape[1]:] = last

    ds = np.hstack((age,ds))

    return ds

fname = "Data/"
df_train = pd.read_csv(fname + "train_features_Sven.csv")
df_label = pd.read_csv(fname + "train_labels_sorted.csv")
df_test = pd.read_csv(fname + "test_features_Sven.csv")

# Preprocessing the data
# mmml + norm
X = pre_pros(df_train)
T = pre_pros(df_test)

#Setting the y to the Subtasks
y1 = df_label[TASK1].to_numpy()
y2 = df_label[TASK2].to_numpy()
y2 = np.ravel(y2)
y3 = df_label[TASK3].to_numpy()

GBC = GradientBoostingClassifier(random_state=12)
GBR = GradientBoostingRegressor(random_state=9)

# TASK 1
print(f'Subtask 1')
pred_1 = np.empty((T.shape[0],len(TASK1)))

for i in range(y1.shape[1]):
    print(f'Round {i+1}: {TASK1[i]}')
    GBC.fit(X, y1[:, i])
    pred_1[:, i] = GBC.predict_proba(T)[:, 1]

# TASK 2
print(f'Subtask 2: {TASK2[0]}')
pred_2 = np.empty((T.shape[0],len(TASK2)))

GBC.fit(X,y2)
pred_2 = GBC.predict_proba(T)[:,1]
pred_2 = pred_2.reshape((pred_2.shape[0],1))

# TASK 3
print(f'Subtask 3')
pred_3 = np.empty((T.shape[0],len(TASK3)))

for i in range(y3.shape[1]):
    print(f'Round {i+1}: {TASK3[i]}')
    GBR.fit(X, y3[:, i])
    pred_3[:, i] = GBR.predict(T)
    
# Merging the predictions of the subtasks and saving to csv

pids = df_test["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

final = np.hstack((pred_1,pred_2))
final = np.hstack((final,pred_3))
final = np.hstack((pids,final))

sub = pd.DataFrame(final,columns=['pid'] + TASK1 + TASK2 + TASK3)
sub = sub.astype({'pid':'int32'})

sub.to_csv('Data/Sub_SvDa.csv', index=False, float_format='%.3f', header=True)
