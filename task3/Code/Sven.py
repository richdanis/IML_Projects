#imports
import numpy as np
from numpy import random as ran
#import pytorch as torch

#data
fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=int)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=int)
print(train)

y = ran.choice(a=[0, 1], size= (train.shape[0],1))
print(y)
for i in range(train.shape[0]):
    if y[i] == 0:
        tmp = train[i,1]
        train[i,1] = train[i,2]
        train[i,2] = tmp
        
print(train)