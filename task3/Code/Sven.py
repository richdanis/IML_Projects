#imports
import numpy as np
#import torch
import torchvision.transforms as T
from PIL import Image

def turn_to_tensorset():

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((250, 350)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),

    ])
    sa = np.empty((10000,3,250,350))
    for i in range(10000):

        name = str(i)
        if i < 10:
            name = '0000' + name
        elif i < 100:
            name = '000' + name
        elif i < 1000:
            name = '00' + name
        else:
            name = '0' + name
        filename = '../Data/food/' + name + '.jpg'
        im1 = Image.open(filename)
        im1 = transform(im1)
        im = im1.numpy()
        sa[i] = im
        if i%50 == 0:
            print(f'Round: {i}')
    
    print(f'Before Save')    
    np.save('../Data/data_as_np.npy',sa)
    print(f'After Save')

#data
#fname = '../Data/'
#food = fname + 'food/'
#train = np.loadtxt(fname + "train_triplets.txt", dtype=str)

turn_to_tensorset()

#k = np.load('../Data/data_as_np.npy')