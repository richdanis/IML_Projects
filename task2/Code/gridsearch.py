import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

train_df, label_df = hf.remove_sparse(train_df, label_df)
train_df, label_df = hf.remove_outliers(train_df, label_df)

train_df2 = train_df.fillna(0)
train_df3 = train_df.fillna(train_df.mean())

X2 = hf.min_mean_max(train_df2)
X3 = hf.min_mean_max(train_df3)
X3 = hf.normalize(X3)

# PREPARING TRAINING LABELS
task2 = label_df.loc[:,"LABEL_Sepsis"]
columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

task3 = label_df[columns].to_numpy()

y2 = task2.to_numpy()
y3 = task3.to_numpy()

params = {'n_estimators':[80,100,120,150,200],'max_depth':[3,5,7]}

gb2 = GradientBoostingClassifier()
clf2 = GridSearchCV(gb2, params, scoring='roc_auc')

clf2.fit(X2, y2)
print("test2")
print(clf2.cv_results_)

for i in range(4):
    gb3 = GradientBoostingRegressor()
    clf3 = GridSearchCV(gb3, params, scoring='r2')
    clf3.fit(X3, y3[:, i])
    print(columns[i])
    print(clf3.cv_results_)
