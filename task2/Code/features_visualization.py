import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as hf

fname = "Data/"
train_features = pd.read_csv(fname + "train_features.csv")
train_labels = pd.read_csv(fname + "train_labels.csv")
test_features = pd.read_csv(fname + "test_features.csv")

# some pids are missing

pd.options.display.max_columns = 40

train_features, train_labels = hf.remove_sparse(train_features,train_labels)
train_features, train_labels = hf.remove_outliers(train_features, train_labels)

#print(train_features.info())
#train_features = train_features.sort_values(by=['pid','Time'])
#print(train_features.head(20))
#train_features = train_features.drop(columns='Time')

#print(len(train_features.columns))
#print(train_features.count())
#print(train_features.head(10))
#train_features = train_features.fillna(train_features.mean())
#print(train_features.head(10))

#missing = train_features.isnull().sum()
#missing = missing[missing > 0]
#missing.sort_values(inplace=True)
#missing.plot.bar()

#sns.distplot(train_features['Time'])


#corr = train_features.corr()
#fig = plt.figure(figsize=(12, 9))
#sns.heatmap(corr, vmax=.8, square=True)
#plt.savefig('features_correlation.png')
#plt.show()

# HCO3 can probably be dropped, really similar to BaseExcess
#

for i in range(len(train_features.columns)):
    train_features[train_features.columns[i]].hist(bins=50)
    plt.title(train_features.columns[i])
    #plt.savefig('features_correlation.png')
    plt.show()






