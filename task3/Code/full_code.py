# %%
# Imports
import numpy as np
from numpy import random as ran

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
import torchvision.models as models
import torchmetrics as metrics
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


def get_ima(train, i):

    A, B, C = train[i, :]

    filename = '../Data/food/'
    A_im = Image.open(filename + A + '.jpg').convert('RGB')
    B_im = Image.open(filename + B + '.jpg').convert('RGB')
    C_im = Image.open(filename + C + '.jpg').convert('RGB')

    resize = T.Resize(size=(250, 350))
    A_im = resize(A_im)
    B_im = resize(B_im)
    C_im = resize(C_im)

    to_tensor = T.ToTensor()
    A_ten = to_tensor(A_im)
    B_ten = to_tensor(B_im)
    C_ten = to_tensor(C_im)

    return A_ten, B_ten, C_ten


def image_to_tensor(img):

    filename = '../Data/food/' + str(img) + '.jpg'
    image = Image.open(filename)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)
    resize = transforms.Resize((250, 350))
    tensor = resize(tensor)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tensor = normalize(tensor)

    return tensor


def get_batch(idx, train):
    batch = torch.empty((64, 3, 3, 250, 350))

    for i in range(64):
        batch[i][0] = image_to_tensor(train[idx * 64 + i][0])
        batch[i][1] = image_to_tensor(train[idx * 64 + i][1])
        batch[i][2] = image_to_tensor(train[idx * 64 + i][2])

    return batch


# %%
vgg16 = models.vgg16(pretrained=True)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        self.classifier[6] = nn.Linear(4096, 1, bias=True)
        self.classifier[0] = nn.Linear(75264, 4096, bias=True)
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    # x represents our data
    def forward(self, x):

        a = self.features(x[0])
        a = self.avgpool(a)
        b = self.features(x[1])
        b = self.avgpool(b)
        c = self.features(x[2])
        c = self.avgpool(c)

        a = torch.cat((a, b, c))
        a = torch.flatten(a)
        a = self.classifier(a)
        a = torch.sigmoid(a)

        return a

    def forwardBatch(self, x):

        res = torch.empty((x.shape[0],))

        for i in range(x.shape[0]):

            a = self.features(x[i][0])
            a = self.avgpool(a)
            b = self.features(x[i][1])
            b = self.avgpool(b)
            c = self.features(x[i][2])
            c = self.avgpool(c)
            a = torch.cat((a, b, c))
            a = torch.flatten(a)
            a = self.classifier(a)
            a = torch.sigmoid(a)
            res[i] = a

        return res


model = Net()#.cuda()
loss_fn = torch.nn.BCELoss()

# define utility functions to compute classification accuracy and
# perform evaluation / testing


#def accuracy(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # computes the classification accuracy
    #correct_label = torch.argmax(pred, axis=-1) == torch.argmax(label, axis=-1)
    #assert correct_label.shape == (pred.shape[0],)
    #acc = torch.mean(correct_label.float())
    #assert 0. <= acc <= 1.
    #return acc

def accuracy(pred, label):
    pred = torch.round(pred)
    correct = 0
    for i in range(pred.shape[0]):
        if pred[i] == label[i]:
            correct += 1
    return correct / pred.shape[0]


def evaluate(model: torch.nn.Module, test) -> torch.Tensor:
    # goes through the test dataset and computes the test accuracy
    model.eval()  # bring the model into eval mode
    with torch.no_grad():
        acc_cum = 0.0
        num_eval_samples = 0
        X = test[:, :3]
        y = test[:, 3:]
        for i in range(test.shape[0] // 64 + 1):
            x_batch_test = get_batch(i, X[i * 64:(i + 1) * 64])
            y_label_test = np.array(y[i * 64:(i + 1) * 64],dtype=float)
            y_label_test = torch.from_numpy(y_label_test)
            #x_batch_test, y_label_test = x_batch_test.cuda(), y_label_test.cuda()
            x_batch_test, y_label_test = x_batch_test, y_label_test
            batch_size = x_batch_test.shape[0]
            num_eval_samples += batch_size
            acc_cum += accuracy(model(x_batch_test), y_label_test) * batch_size
        avg_acc = acc_cum / num_eval_samples
        avg_acc = torch.tensor(avg_acc)
        assert 0 <= avg_acc <= 1
        return avg_acc


# %%
# Setup the optimizer (adaptive learning rate method)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
train = swap(train)

np.random.shuffle(train)

val = train[int(0.8*train.shape[0]):]
train = train[:int(0.8*train.shape[0])]

for epoch in range(50):
    # reset statistics trackers
    train_loss_cum = 0.0
    acc_cum = 0.0
    num_samples_epoch = 0
    t = time.time()
    np.random.shuffle(train)
    X = train[:, :3]
    y = train[:, 3:]

    # Go once through the training dataset (-> epoch)
    for i in range(train.shape[0] // 64 + 1):
        x_batch = get_batch(i, X[i * 64:(i + 1) * 64])
        y_batch = np.array(y[i * 64:(i + 1) * 64],dtype=float)
        y_batch = torch.from_numpy(y_batch)

        # zero grads and put model into train mode
        optim.zero_grad()
        model.train()

        # move data to GPU
        #x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_batch, y_batch = x_batch, y_batch

        # forward pass
        pred = model.forwardBatch(x_batch)
        loss = loss_fn(pred, y_batch)

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
    test_acc = evaluate(model, val)
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

