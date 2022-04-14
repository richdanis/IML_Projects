import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import helper_functions as hf
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

fname = "Data/"
train_df = pd.read_csv(fname + "train_features.csv")
label_df = pd.read_csv(fname + "train_labels.csv")

train_df = train_df[["Time","pid","RRate","ABPm","SpO2","Heartrate"]]

#X = hf.take_mean_features(train_df,["Time"])
#X = hf.normalize(X)

columns = ["LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"]

y = label_df[columns].to_numpy()

columns = ["RRate","ABPm","SpO2","Heartrate"]

for i in range(4):
    reg = SGDRegressor(max_iter=10000)
    X = hf.take_mean_features(train_df[["Time","pid",columns[i]]],["Time"])
    scores = cross_val_score(reg, X, y[:, i], cv=10)
    print("Score: ",np.mean(scores))

'''
Score:  0.37592154179722764
Score:  0.5728583008037672
Score:  0.2578932417363745
Score:  0.596710935341267
'''