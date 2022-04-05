import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fname = "/home/richard/Documents/ETH 6.Semester/Introduction to Machine Learning/task2_k49am2lqi/"
train_features = pd.read_csv(fname + "train_features.csv")
train_labels = pd.read_csv(fname + "train_labels.csv")
test_features = pd.read_csv(fname + "test_features.csv")

# some pids are missing

pd.options.display.max_columns = 6

#print(train_features.info())
#train_features = train_features.sort_values(by=['pid','Time'])
#print(train_features.head(20))
#train_features = train_features.drop(columns='Time')
#print(train_features.describe())
#print(train_features.head(10))
#train_features = train_features.fillna(train_features.mean())
#print(train_features.head(10))

missing = train_features.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
#missing.plot.bar()
#sns.distplot(train_features['Time'])
corr = train_features.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)
plt.savefig('correlation.png')
plt.show()

# HCO3 can probably be dropped, really similar to BaseExcess
#







