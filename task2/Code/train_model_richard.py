import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fname = "Data/"
train_features = pd.read_csv(fname + "train_features.csv")
train_labels = pd.read_csv(fname + "train_labels.csv")
test_features = pd.read_csv(fname + "test_features.csv")

train_features = train_features.sort_values(by=['pid','Time'])
train_features = train_features.drop(columns='Time')
train_features = train_features.drop(columns='HCO3')
train_features = train_features.drop(columns='ABPm')

train_features = train_features.fillna(train_features.mean())
train_features = train_features.to_numpy()
pids = np.unique(train_features[:,0])
print(len(pids))
# have to somehow exclude age?

train_features = train_features[:,1:]
train_features = train_features.reshape((len(pids),12*train_features.shape[-1]))







