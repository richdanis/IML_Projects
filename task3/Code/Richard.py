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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tensor = normalize(tensor)

    return tensor


def get_batch(train,begin):

    batch = torch.empty((train.shape[0],3,3,250,350))

    for i in range(train.shape[0]):

        batch[i][0] = image_to_tensor(train[begin+i][0])
        batch[i][1] = image_to_tensor(train[begin+i][1])
        batch[i][2] = image_to_tensor(train[begin+i][2])

    return batch

#data
fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=int)
#print("hey")

batch = get_batch(train[:64],0)
print("hey")
#filename = '../Data/food/' + str(train[0][0]) + '.jpg'
#img = Image.open(filename)
#to_tensor = transforms.ToTensor()
#tensor = to_tensor(img)
#resize = transforms.Resize((250, 350))
#tensor = resize(tensor)

#test = ImageToTensor(train[0][0])
#print("hehe")





