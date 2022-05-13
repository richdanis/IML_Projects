from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import time


class TrainDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx]).float()
        return feature


class MolecularNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1000, 900),
            nn.LeakyReLU(),
            nn.Linear(900, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 100),
        )

        self.regressor = nn.Sequential(
            nn.Linear(100, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.regressor(x)

        return x


def get_loaders(dataset, batch_size=64, shuffle=True):
    features = pd.read_csv("Data/" + dataset + "_features.csv")

    features = features.drop(columns=['Id', 'smiles'])

    features = features.to_numpy()

    labels = pd.read_csv("Data/" + dataset + "_labels.csv")

    labels = labels.drop(columns=['Id'])

    labels = labels.to_numpy()

    combined = np.hstack((features, labels))

    if shuffle:
        np.random.shuffle(combined)

    split = int(0.8 * combined.shape[0])

    train = combined[:split]
    val = combined[split:]

    print(train.shape[0])
    print(val.shape[0])

    full_dataset = TrainDataset(combined)
    train_dataset = TrainDataset(train)
    validation_dataset = TrainDataset(val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    return train_loader, val_loader, full_loader


def evaluate(models, loss_fn, val_loader, has_label, device):
    # goes through the test dataset and computes the test accuracy
    val_loss_cum = 0.0

    # bring the models into eval mode
    for model in models:
        model.eval()
    y_batch_val = None

    with torch.no_grad():
        num_eval_samples = 0
        for x_batch_val in val_loader:

            if has_label:
                y_batch_val = x_batch_val[:, -1]
                y_batch_val = torch.reshape(y_batch_val, (y_batch_val.shape[0], 1))
                y_batch_val = y_batch_val.to(device)

            x_batch_val = x_batch_val[:, :-1].to(device)
            x_val = x_batch_val

            for model in models:
                x_val = model(x_val)

            loss = None

            if has_label:
                loss = loss_fn(x_val, y_batch_val)
            else:
                loss = loss_fn(x_val, x_batch_val)

            num_samples_batch = x_batch_val.shape[0]
            num_eval_samples += num_samples_batch
            val_loss_cum += loss * num_samples_batch

        avg_val_loss = val_loss_cum / num_eval_samples
        return avg_val_loss


def train_loop(models, train_loader, val_loader, loss_fn, optim, device, has_label, show=1, save=40, epochs=200):
    for epoch in range(epochs):
        # reset statistics trackers
        train_loss_cum = 0.0
        mse_loss_cum = 0.0
        num_samples_epoch = 0
        y_batch = None
        t = time.time()
        # Go once through the training dataset (-> epoch)

        for x_batch in train_loader:

            if has_label:
                y_batch = x_batch[:, -1]
                y_batch = torch.reshape(y_batch, (y_batch.shape[0], 1))
                y_batch = y_batch.to(device)

            # move data to GPU
            x_batch = x_batch[:, :-1]
            x_batch = x_batch.to(device)

            # zero grads and put model into train mode
            optim.zero_grad()
            models[-1].train()

            x = x_batch

            # forward pass
            for model in models:
                x = model(x)

            # loss
            loss = None

            if has_label:
                loss = loss_fn(x, y_batch)
            else:
                loss = loss_fn(x, x_batch)

            # backward pass and gradient step
            loss.backward()
            optim.step()

            # keep track of train stats
            num_samples_batch = x_batch.shape[0]
            num_samples_epoch += num_samples_batch
            train_loss_cum += loss * num_samples_batch

        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_train_loss = torch.sqrt(avg_train_loss)
        val_loss = evaluate(models, loss_fn, val_loader, has_label, device)
        val_loss = torch.sqrt(val_loss)
        epoch_duration = time.time() - t

        # print some infos
        if epoch % show == 0:
            print(f'Epoch {epoch} | Train loss: {avg_train_loss:.4f} | '
                  f' Validation loss: {val_loss:.4f} | '
                  f' Duration {epoch_duration:.2f} sec')

        # save checkpoint of model
        if (epoch % save == 0) and epoch > 0:
            save_path = f'model_epoch_{epoch}.pt'
            torch.save(models[-1],
                       save_path)
            print(f'Saved model checkpoint to {save_path}')