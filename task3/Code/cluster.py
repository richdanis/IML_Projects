#Imports
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


def swap(train):
    y = ran.choice(a=[0, 1], size=(train.shape[0], 1))

    for i in range(train.shape[0]):
        if y[i] == 0:
            tmp = train[i, 1]
            train[i, 1] = train[i, 2]
            train[i, 2] = tmp

    train = np.hstack((train, y))

    return train

class ImgDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((250, 350)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),

        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        filename = 'Data/food/' + str(self.data[idx][0]) + '.jpg'
        im1 = Image.open(filename)
        im1 = self.transform(im1)
        filename = 'Data/food/' + str(self.data[idx][1]) + '.jpg'
        im2 = Image.open(filename)
        im2 = self.transform(im2)
        filename = 'Data/food/' + str(self.data[idx][2]) + '.jpg'
        im3 = Image.open(filename)
        im3 = self.transform(im3)

        label = torch.tensor(float(self.data[idx][3]))
        label = label.float()

        im1 = torch.unsqueeze(im1, 0)
        im2 = torch.unsqueeze(im2, 0)
        im3 = torch.unsqueeze(im3, 0)

        return torch.cat((im1,im2,im3)),label


#vgg16 = torch.load('./vgg16.pth')
pretrained = torch.load('./efficientnet_b3.pth')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = pretrained.features
        self.avgpool = pretrained.avgpool  # output 512x7x7 = 25'088 output nodes
        self.classifier = pretrained.classifier
        self.classifier[1] = nn.Linear(1536 * 3, 1, bias=True)
        #self.classifier[0] = nn.Linear(25088 * 3, 4096, bias=True)
        #self.classifier[6] = nn.Linear(4096, 1, bias=True)

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


model = Net().cuda()

loss_fn = torch.nn.BCEWithLogitsLoss()

# define utility functions to compute classification accuracy and
# perform evaluation / testing

def accuracy(pred, label):
    label = label.int().cuda()
    accur = Accuracy().cuda()
    acc = accur(pred, label)
    return acc

def evaluate(model: torch.nn.Module, testloader) -> torch.Tensor:
    # goes through the test dataset and computes the test accuracy
    model.eval()  # bring the model into eval mode
    with torch.no_grad():
        acc_cum = 0.0
        num_eval_samples = 0

        for x_batch_test, y_label_test in testloader:
            y_label_test = torch.reshape(y_label_test,(y_label_test.shape[0],1))
            x_batch_test, y_label_test = x_batch_test.cuda(), y_label_test.cuda()
            batch_size = x_batch_test.shape[0]
            num_eval_samples += batch_size
            acc_cum += accuracy(model(x_batch_test), y_label_test) * batch_size

        avg_acc = acc_cum / num_eval_samples
        avg_acc = torch.tensor(avg_acc)
        assert 0 <= avg_acc <= 1
        return avg_acc


# Setup the optimizer (adaptive learning rate method)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

fname = 'Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
train = swap(train)

np.random.shuffle(train)

val = train[int(0.8 * train.shape[0]):]
train = train[:int(0.8 * train.shape[0])]

train_dataset = ImgDataset(train)
val_dataset = ImgDataset(val)

trainloader = DataLoader(train_dataset, batch_size=20,
                         shuffle=True, num_workers=0)

valloader = DataLoader(val_dataset, batch_size=20,
                       shuffle=True, num_workers=0)

for epoch in range(50):
    # reset statistics trackers
    train_loss_cum = 0.0
    acc_cum = 0.0
    num_samples_epoch = 0
    t = time.time()

    # Go once through the training dataset (-> epoch)
    for x_batch, y_batch in trainloader:

        # zero grads and put model into train mode
        optim.zero_grad()
        model.train()

        y_batch = torch.reshape(y_batch, (y_batch.shape[0], 1))

        # move data to GPU
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        # forward pass

        # pred = model.forwardBatch(x_batch)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        # loss = my_loss()

        # backward pass and gradient step
        loss.backward()
        optim.step()

        # keep track of train stats
        num_samples_batch = x_batch.shape[0]
        num_samples_epoch += num_samples_batch
        train_loss_cum += loss * num_samples_batch
        acc_cum += accuracy(pred, y_batch) * num_samples_batch


    # average the accumulated statistics
    avg_train_loss = train_loss_cum / num_samples_epoch
    avg_acc = acc_cum / num_samples_epoch
    test_acc = evaluate(model, valloader)
    epoch_duration = time.time() - t

    # print some infos
    print(f'Epoch {epoch} | Train loss: {train_loss_cum:.4f} | '
          f' Train accuracy: {avg_acc:.4f} | Test accuracy: {test_acc.item():.4f} |'
          f' Duration {epoch_duration:.2f} sec')

    # save checkpoint of model
    if epoch % 5 == 0 and epoch > 0:
        save_path = f'model_epoch_{epoch}.pt'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()},
                   save_path)
        print(f'Saved model checkpoint to {save_path}')


