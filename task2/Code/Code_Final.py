import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

# drop pid, time and age column
# then for each patient, take the min, mean, max and last of all feature columns
# results in 34 * 4 = 136 features per patient
# add age back, 137 features per patient

def min_mean_max_last(df):

    X = df.to_numpy()
    # exclude pids age and time
    age = X[::12, 0]
    age = age.reshape((len(age), 1))
    X = X[:, 3:]

    ds = np.empty((X.shape[0]//12,X.shape[1]*4))

    for i in range(ds.shape[0]):
        mean = X[(i*12):((i+1)*12)].mean(axis=0)
        min = X[(i*12):((i+1)*12)].min(axis=0)
        max = X[(i*12):((i+1)*12)].max(axis=0)
        last = X[(i+1) * 12 - 1]
        ds[i,:X.shape[1]] = mean
        ds[i,X.shape[1]:2*X.shape[1]] = min
        ds[i,2*X.shape[1]:3*X.shape[1]] = max
        ds[i,3*X.shape[1]:] = last

    ds = np.hstack((age,ds))

    return ds

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

# Preprocessing the data

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
X = min_mean_max_last(train_df)
T = min_mean_max_last(test_df)

TASK1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TASK2 = ['LABEL_Sepsis']
TASK3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

y1 = label_df[TASK1].to_numpy()
y2 = label_df.loc[:,"LABEL_Sepsis"].to_numpy()
y3 = label_df[TASK3].to_numpy()

GBC = GradientBoostingClassifier(random_state=12)
GBR = GradientBoostingRegressor(random_state=9)

# TASK 1

pred1 = np.empty((T.shape[0],len(TASK1)))

for i in range(y1.shape[1]):
    GBC.fit(X, y1[:, i])
    pred1[:, i] = GBC.predict_proba(T)[:, 1]

# TASK 2

GBC.fit(X,y2)
pred2 = GBC.predict_proba(T)[:,1]
pred2 = pred2.reshape((pred2.shape[0],1))

# TASK 3

pred3 = np.empty((T.shape[0],len(TASK3)))

for i in range(y3.shape[1]):
    GBR.fit(X, y3[:, i])
    pred3[:, i] = GBR.predict(T)

# Merging the predictions of the subtask and saving to csv

pids = test_df["pid"].to_numpy()[::12]
pids = pids.reshape((len(pids),1))

final = np.hstack((pred1,pred2))
final = np.hstack((final,pred3))
final = np.hstack((pids,final))

sub = pd.DataFrame(final,columns=['pid'] + TASK1 + TASK2 + TASK3)
sub = sub.astype({'pid':'int32'})
sub.to_csv('Data/Final_submission.csv', index=False, float_format='%.3f', header=True)


