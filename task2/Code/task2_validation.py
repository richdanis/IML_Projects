import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import helper_functions as hf

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")
test_df = pd.read_csv(fname + "test_features.csv")

train_df, label_df = hf.remove_outliers(train_df, label_df)

X = hf.prepare_dataset(train_df,["Time"])

# PREPARING TRAINING LABELS
label_df = label_df.loc[:,"LABEL_Sepsis"]

y = label_df.to_numpy()

clf = SGDClassifier()
X_new = SelectKBest(k=61).fit_transform(X, y)
scores = cross_val_score(clf, X_new, y, cv=10)
print(np.mean(scores))
