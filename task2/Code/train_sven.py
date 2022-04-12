#1
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest

#2
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', None)

#3
train_str = 'data_interpolated_means.csv'
labels_str = 'labels_sorted.csv'
test_str = 'test_features.csv'

df_train = pd.read_csv(train_str)
df_labels = pd.read_csv(labels_str)
df_test = pd.read_csv(test_str)

#4
print(df_train.shape)
df = df_train.copy()

#5
df_labels = df_labels.sort_values(by=['pid'])

TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TEST2 = ['LABEL_Sepsis']
TEST3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

df_subtask1 = df_labels[TEST1]
df_subtask2 = df_labels[TEST2]
df_subtask3 = df_labels[TEST3]

#6
df_np = df.copy().to_numpy()


#7
df_try = df.copy()
# PREPARING TRAINING FEATURES
df_try = df_try.drop(columns=['Time'])

# fill NaN with median for each column
df_try=(df_try-df_try.mean())/df_try.std()

# convert to numpy array
X = df_try.to_numpy()
pids = np.unique(X[:,0])

# exclude pids
X = X[:,1:]
age = X[::12,0]
age = age.reshape((len(age),1))

# exclude ages
X = X[:,1:]
X = X.reshape((len(pids),12*X.shape[-1]))
X = np.hstack((age,X))

#poly = PolynomialFeatures(10)

# PREPARING TRAINING LABELS

y = df_subtask1.copy().to_numpy()
y = y[:,1:]

#accuracy = np.empty((y.shape[1],))

clf = SGDClassifier()
for i in range(y.shape[1]):
    print("label: " + TEST1[i])
    X_new = SelectKBest(k=409).fit_transform(X, y[:,i])
    #X_new = poly.fit_transform(X_new)
    scores = cross_val_score(clf, X_new, y[:,i], cv=10)
    print(scores)

#for i in range(y.shape[1]):
    #clf = LogisticRegressionCV(cv=10, max_iter=100, solver='sag').fit(X,y[:,i])
    #accuracy[i] = clf.score(X,y)