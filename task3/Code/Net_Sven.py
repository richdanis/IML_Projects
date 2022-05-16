#Imports
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

import time
from PIL import Image

class ImgDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((250, 350)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.iter = 6000

    def __len__(self):
        return self.iter

    def __getitem__(self, idx):

        filename = '../Data/food/' + str(self.data[idx][0]) + '.jpg'
        im1 = Image.open(filename)
        im1 = self.transform(im1)
        filename = '../Data/food/' + str(self.data[idx][1]) + '.jpg'
        im2 = Image.open(filename)
        im2 = self.transform(im2)
        filename = '../Data/food/' + str(self.data[idx][2]) + '.jpg'
        im3 = Image.open(filename)
        im3 = self.transform(im3)

        im1 = torch.unsqueeze(im1, 0)
        im2 = torch.unsqueeze(im2, 0)
        im3 = torch.unsqueeze(im3, 0)

        return torch.cat((im1,im2,im3))
    

cn = models.convnext_tiny(pretrained=True)

class CNNet(nn.Module):

    def __init__(self):
        super(CNNet, self).__init__()
        self.features = cn.features
        self.avgpool = cn.avgpool

    def forward(self, x):
        # [Batch_Size, 3, 3, 250, 350]
        sh1 = ((x.shape[0]*3,) + x[0][0].shape)
        x = torch.reshape(x,sh1)
        # [3*Batch_Size, 3, 250, 350]
        x = self.features(x)  
        # [3*Batch_Size, 768, 7, 10]
        x = self.avgpool(x)
        # [3*Batch_Size, 768, 1, 1]
        x = torch.flatten(x, start_dim=1)
        # [3*Batch_Size, 768]
        sh = (int(x.shape[0]//3),int(3),int(x.shape[-1]))
        x = torch.reshape(x,sh)
        # [3*Batch_Size, 3, 768]
        return x

model = CNNet()

# Loss function 
loss_2 = torch.nn.TripletMarginLoss(margin=0.1, reduce=False)
# p=2.0, eps=1e-06, swap=False, size_average=None, reduction='mean'

# define utility functions to compute classification accuracy and
# perform evaluation / testing
def accuracy(label):
    return torch.sum(label<=0.1) / label.shape[0]

def evaluate(model: torch.nn.Module) -> torch.Tensor:
    # goes through the test dataset and computes the test accuracy
    model.eval()  # bring the model into eval mode
    with torch.no_grad():
        acc_cum = 0.0
        num_eval_samples = 0
        for x_batch_test in test_loader:
            batch_size = x_batch_test.shape[0]
            num_eval_samples += batch_size
            pred = model(x_batch_test) 
            loss = loss_2(pred[:,0,:],pred[:,1,:],pred[:,2,:])
            acc_cum += accuracy(loss) * batch_size

        avg_acc = acc_cum / num_eval_samples
        avg_acc = torch.tensor(avg_acc)
        assert 0 <= avg_acc <= 1
        return avg_acc
    

fname = '../Data/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)

np.random.shuffle(train)

val = train[int(0.8*train.shape[0]):]
train = train[:int(0.8*train.shape[0])]

train_dataset = ImgDataset(train)
val_dataset = ImgDataset(val)

train_loader = DataLoader(train_dataset, batch_size=20, num_workers=0,
                          shuffle=True, pin_memory=True) 
test_loader = DataLoader(val_dataset, batch_size=20, num_workers=0,
                         shuffle=True, pin_memory=True)

# Setup the optimizer (adaptive learning rate method)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    # reset statistics trackers
    train_loss_cum = 0.0
    acc_cum = 0.0
    num_samples_epoch = 0
    t = time.time()

    # Go once through the training dataset (-> epoch)
    for x_batch in train_loader:

        # zero grads and put model into train mode
        optim.zero_grad()
        model.train()

        # forward pass
        logits = model(x_batch) #[batch_size,3,768]
        loss = loss_2(logits[:,0,:],logits[:,1,:],logits[:,2,:])

        # backward pass and gradient step
        torch.sum(loss).backward()
        optim.step()
        
        # keep track of train stats
        num_samples_batch = x_batch.shape[0]
        num_samples_epoch += num_samples_batch
        train_loss_cum += torch.sum(loss) * num_samples_batch
        acc_cum += accuracy(loss) * num_samples_batch
        print('hi')
        #torch.cuda.empty_cache()

    # average the accumulated statistics
    avg_train_loss = train_loss_cum / num_samples_epoch
    avg_acc = acc_cum / num_samples_epoch
    test_acc = evaluate(model)
    epoch_duration = time.time() - t

    # print some infos
    print(f'Epoch {epoch} | Train loss: {train_loss_cum:.4f} | '
          f' Train accuracy: {avg_acc:.4f} | Test accuracy:{test_acc}  |'  #{test_acc.item():.4f}
          f' Duration {epoch_duration:.2f} sec')
    
    # save checkpoint of model
    if epoch % 5 == 0 and epoch > 0:
        save_path = f'model_epoch_{epoch}.pt'
        torch.save(model, save_path)
        print(f'Saved model checkpoint to {save_path}')