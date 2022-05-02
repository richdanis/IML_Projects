import torch

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import help_Sven as hs


def image_to_tensor(img):

    filename = '../Data/food/' + str(img) + '.jpg'
    image = Image.open(filename)
    to_tensor = T.ToTensor()
    tensor = to_tensor(image)
    resize = T.Resize((250,350))
    tensor = resize(tensor)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tensor = normalize(tensor)

    return tensor


def get_batch(train,begin):

    batch = torch.empty((train.shape[0],3,3,250,350))

    for i in range(train.shape[0]):

        batch[i][0] = image_to_tensor(train[i][0])
        batch[i][1] = image_to_tensor(train[i][1])
        batch[i][2] = image_to_tensor(train[i][2])

    return batch

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
        filename = '../Data/food/' + str(self.data[idx][0]) + '.jpg'
        im1 = Image.open(filename)
        im1 = self.transform(im1)
        filename = '../Data/food/' + str(self.data[idx][1]) + '.jpg'
        im2 = Image.open(filename)
        im2 = self.transform(im2)
        filename = '../Data/food/' + str(self.data[idx][2]) + '.jpg'
        im3 = Image.open(filename)
        im3 = self.transform(im3)

        label = int(self.data[idx][3])

        im1 = torch.unsqueeze(im1, 0)
        im2 = torch.unsqueeze(im2, 0)
        im3 = torch.unsqueeze(im3, 0)

        return torch.cat((im1,im2,im3)),label



#data
fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
#test = np.loadtxt(fname + "test_triplets.txt", dtype=int)
#print("hey")

train = hs.swap(train)
dataset = ImgDataset(train)

dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=0)

for x_batch,y_batch in dataloader:
    print(x_batch.shape[0])
    print(y_batch.shape[0])

#batch = get_batch(train[:64],0)
#print("hey")
#filename = '../Data/food/' + str(train[0][0]) + '.jpg'
#img = Image.open(filename)
#to_tensor = transforms.ToTensor()
#tensor = to_tensor(img)
#resize = transforms.Resize((250, 350))
#tensor = resize(tensor)

#test = ImageToTensor(train[0][0])
#print("hehe")





