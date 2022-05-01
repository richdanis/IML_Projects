#imports
import numpy as np
from numpy import random as ran
import help_Sven as h
import torch
import Richard as R

#data
fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=str)

train = h.swap(train)
i = 1000
A,B,C = h.get_ima(train,i)

K = R.image_to_tensor('00905')

print(A.shape)
print(torch.min(A))
print(torch.max(A))
print(torch.min(K))
print(torch.max(K))
#print(B.shape)
#print(C.shape)
print('Hello World!')