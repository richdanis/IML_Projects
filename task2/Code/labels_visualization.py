import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fname = "Data/"

pd.options.display.max_columns = 40

train_labels = pd.read_csv(fname + "train_labels.csv")
train_labels = train_labels.drop(columns=['pid','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate'])

print(train_labels.describe())
#print(train_labels.head(100))
#train_labels = train_labels.sort_values(by=['pid'])

counts = []
columns = []
for column in train_labels:
    columns.append(column)
    counts.append(train_labels[column].value_counts()[1])

y_pos = np.arange(len(columns))
fig = plt.figure()
plt.bar(y_pos, counts)
plt.title("Number of tests per label")
plt.savefig("Plots/tests_per_category.png")
plt.show()

rowSums = train_labels.iloc[:,:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
plt.bar(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Persons having multiple (or zero) labels")
plt.tight_layout()
plt.savefig("Plots/persons_having_multiple_labels.png")
plt.show()

corr = train_labels.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)
plt.title("Correlation of labels, only task 1")
plt.savefig('Plots/label_correlation.png')
plt.show()


