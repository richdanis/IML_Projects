import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from Richard import image_to_tensor


vgg16 = models.vgg16(pretrained=True)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        self.classifier[6] = nn.Linear(4096,1,bias=True)
        self.classifier[0] = nn.Linear(75264,4096,bias=True)

    # x represents our data
    def forward(self, x):

        a = self.features(x[0])
        a = self.avgpool(a)
        b = self.features(x[1])
        b = self.avgpool(b)
        c = self.features(x[2])
        c = self.avgpool(c)

        a = torch.cat((a,b,c))
        a = torch.flatten(a)

        return self.classifier(a)

a = image_to_tensor('09896')
b = image_to_tensor('09640')
c = image_to_tensor('09177')

test = Net()
res = test.forward((a,b,c))
print("hey")