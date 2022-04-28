#imports
import numpy as np
from numpy import random as ran
import help_Sven as h
import torch

#data
fname = 'Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=int)

y, train = h.swap(train)
i = 1000

A,B,C = h.get_ima(train,i)

print(A.shape)
print(B.shape)
print(C.shape)

print('Hello World!')
