import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import helper_functions as hf
from sklearn.metrics import roc_auc_score

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

X = hf.min_mean_max(train_df)

# PREPARING TRAINING LABELS
label_df = label_df.loc[:,"LABEL_Sepsis"]

y = label_df.to_numpy()

clf = SGDClassifier()
X_new = SelectKBest(k=20).fit_transform(X, y)
scores = cross_val_score(clf, X_new, y,scoring='roc_auc', cv=10)
print(np.mean(scores))

'''
0.6800140569659741
'''