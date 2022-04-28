#imports
import numpy as np
from numpy import random as ran
import torchvision.transforms as T
from PIL import Image

def swap(train):
    y = ran.choice(a=[0, 1], size= (train.shape[0],1))

    for i in range(train.shape[0]):
        if y[i] == 0:
            tmp = train[i,1]
            train[i,1] = train[i,2]
            train[i,2] = tmp

    return y, train

def get_ima(train,i):
    A,B,C = train[i,:]
    
    food = 'Data/food/'
    A_im = Image.open(food + A + '.jpg').convert('RGB')
    B_im = Image.open(food + B + '.jpg').convert('RGB')
    C_im = Image.open(food + C + '.jpg').convert('RGB')
    
    resize = T.Resize(size = (250, 350))
    A_im = resize(A_im)
    B_im = resize(B_im)
    C_im = resize(C_im)
        
    to_tensor = T.ToTensor()
    A_ten = to_tensor(A_im)
    B_ten = to_tensor(B_im)
    C_ten = to_tensor(C_im)
    
    return A_ten, B_ten, C_ten
