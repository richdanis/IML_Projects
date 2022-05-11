import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import time


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def normalize(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    mean = np.resize(mean, (1, mean.shape[0]))
    std = np.resize(std, (1, std.shape[0]))
    return (X - mean) / std


class FeaturesDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5)

        self.encoder = nn.Sequential(
            nn.Linear(1000, 500),
            self.dropout,
            nn.LeakyReLU(),
            nn.Linear(500, 50),
        )

        self.decoder = nn.Sequential(
            nn.Linear(50, 500),
            self.dropout,
            nn.LeakyReLU(),
            nn.Linear(500, 1000),
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


pretrain_features = pd.read_csv("Data/pretrain_features.csv")
pretrain_features = pretrain_features.drop(columns=['Id', 'smiles'])

pretrain_features = pretrain_features.to_numpy()
pretrain_dataset = FeaturesDataset(pretrain_features)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
# pretrain_features = normalize(pretrain_features)
model = AutoEncoder()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(10):
    # reset statistics trackers
    train_loss_cum = 0.0
    acc_cum = 0.0
    num_samples_epoch = 0
    t = time.time()
    # Go once through the training dataset (-> epoch)
    count = 0
    for x_batch in pretrain_loader:
        # zero grads and put model into train mode
        optim.zero_grad()
        model.train()

        # move data to GPU
        x_batch = x_batch.to(device)

        # forward pass
        x = model(x_batch)

        # loss
        loss = loss_fn(x, x_batch)

        # backward pass and gradient step
        loss.backward()
        optim.step()

        # keep track of train stats
        num_samples_batch = x_batch.shape[0]
        num_samples_epoch += num_samples_batch
        train_loss_cum += loss * num_samples_batch

        # end epoch after 100 batches
        count += 1
        if count == 100:
            break

    # average the accumulated statistics
    avg_train_loss = train_loss_cum / num_samples_epoch
    avg_acc = acc_cum / num_samples_epoch
    test_acc = evaluate(model)
    epoch_duration = time.time() - t

    # print some infos
    print(f'Epoch {epoch} | Train loss: {train_loss_cum:.4f} | '
          f' Duration {epoch_duration:.2f} sec')

    # save checkpoint of model
    if (epoch % 5 == 0 or epoch % 4 == 0 or epoch % 3 == 0 or epoch % 2 == 0) and epoch > 0:
        save_path = f'model_epoch_{epoch}.pt'
        torch.save(model,
                   save_path)
        print(f'Saved model checkpoint to {save_path}')

