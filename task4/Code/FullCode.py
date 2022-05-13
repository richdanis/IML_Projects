import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import time
import random
from task4_lib import *


# fix seeds
torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

pretrain_features = pd.read_csv("Data/pretrain_features.csv")
pretrain_features = pretrain_features.drop(columns=['Id', 'smiles'])
pretrain_features = pretrain_features.to_numpy()

pretrain_labels = pd.read_csv("Data/pretrain_labels.csv")
pretrain_labels = pretrain_labels.drop(columns=['Id'])
pretrain_labels = pretrain_labels.to_numpy()

np.random.shuffle(pretrain_features)

split = int(0.8*pretrain_features.shape[0])

pretrain = pretrain_features[:split]
pretrain_val = pretrain_features[split:]


pretrain_dataset = PretrainDataset(pretrain)
pretrain_validation_dataset = PretrainDataset(pretrain_val)

train_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(pretrain_validation_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)