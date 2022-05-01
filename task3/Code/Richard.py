import torch

import numpy as np
from torchvision import transforms
from PIL import Image


def image_to_tensor(img):

    filename = '../Data/food/' + str(img) + '.jpg'
    image = Image.open(filename)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)
    resize = transforms.Resize((250,350))
    tensor = resize(tensor)

    return tensor


def get_batch(idx, train):

    batch = torch.empty((64,3,250,350))

    for i in range(64):

        batch[i][0] = ImageToTensor(train[idx*64+i][0])
        batch[i][1] = ImageToTensor(train[idx*64+i][1])
        batch[i][2] = ImageToTensor(train[idx*64+i][2])

    return batch





#data
#fname = '../Data/'
#food = fname + 'food/'
#train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=int)
#print("hey")

#filename = '../Data/food/' + str(train[0][0]) + '.jpg'
#img = Image.open(filename)
#to_tensor = transforms.ToTensor()
#tensor = to_tensor(img)
#resize = transforms.Resize((250, 350))
#tensor = resize(tensor)

#test = ImageToTensor(train[0][0])
#print("hehe")





