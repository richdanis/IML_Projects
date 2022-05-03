import numpy as np
from numpy import random as ran

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch._C import dtype

from torchmetrics import Accuracy

import time

from PIL import Image

test = np.loadtxt('Data/test_triplets.txt',dtype=str)

#vgg16 = models.vgg16(pretrained=True)
vgg16 = torch.load('vgg16.pth')
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool  # output 512x7x7 = 25'088 output nodes
        self.classifier = vgg16.classifier
        self.classifier[0] = nn.Linear(25088 * 3, 4096, bias=True)
        self.classifier[6] = nn.Linear(4096, 1, bias=True)
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    # x represents our data
    def forward(self, x):

        with torch.no_grad():
            x = torch.reshape(x, (x.shape[0] * 3,) + x[0][0].shape)

            x = self.features(x)
            x = self.avgpool(x)

            x = torch.reshape(x, (x.shape[0] // 3, 3) + x[0].shape)
            x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x

model = torch.load('vgg_epoch_40.pth').cuda()
model.eval()

transform = T.Compose([
            T.ToTensor(),
            T.Resize((250, 350)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

output = np.empty((test.shape[0],),dtype=int)

for row in range(test.shape[0]):
    
    filename = 'Data/food/' + str(test[row][0]) + '.jpg'
    im1 = Image.open(filename)
    im1 = transform(im1)
    filename = 'Data/food/' + str(test[row][1]) + '.jpg'
    im2 = Image.open(filename)
    im2 = transform(im2)
    filename = 'Data/food/' + str(test[row][2]) + '.jpg'
    im3 = Image.open(filename)
    im3 = transform(im3)

    im1 = torch.unsqueeze(im1, 0)
    im2 = torch.unsqueeze(im2, 0)
    im3 = torch.unsqueeze(im3, 0)

    x = torch.cat((im1, im2, im3))
    x = torch.unsqueeze(x,0).cuda()
    res = model(x)
    res = torch.sigmoid(res)
    res = torch.round(res)

    output[row] = res

np.savetxt('first_submission.txt',output,fmt='%i')


