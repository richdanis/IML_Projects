import torch

import numpy as np
from torchvision import transforms
from PIL import Image

def ImageToTensor(img):

    filename = '../Data/food/' + str(img) + '.jpg'
    image = Image.open(filename)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)
    tensor = tensor.resize((3,))

    return tensor


#data
fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=int)
print("hey")

filename = '../Data/food/' + str(train[0][0]) + '.jpg'
img = Image.open(filename)
to_tensor = transforms.ToTensor()
tensor = to_tensor(img)

filename = '../Data/food/' + str(train[0][1]) + '.jpg'
img = Image.open(filename)
to_tensor = transforms.ToTensor()
tensor = to_tensor(img)
print("hehe")





