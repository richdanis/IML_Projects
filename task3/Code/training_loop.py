import numpy as np
import torch
import time
from Richard import *

# Setup the optimizer (adaptive learning rate method)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

fname = '../Data/'
food = fname + 'food/'
train = np.loadtxt(fname + "train_triplets.txt", dtype=str)
train = swap(train)

for epoch in range(50):
    # reset statistics trackers
    train_loss_cum = 0.0
    acc_cum = 0.0
    num_samples_epoch = 0
    t = time.time()
    np.random.shuffle(train)
    X = train[:,:3]
    y = train[:,3:]

    # Go once through the training dataset (-> epoch)
    for i in range(train.shape[0]/64 + 1):

        x_batch = get_batch(i,X[i*64:(i+1)*64])
        y_batch = y[i*64:(i+1)*64]

        # zero grads and put model into train mode
        optim.zero_grad()
        model.train()

        # move data to GPU
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        # forward pass
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        # backward pass and gradient step
        loss.backward()
        optim.step()

        # keep track of train stats
        num_samples_batch = x_batch.shape[0]
        num_samples_epoch += num_samples_batch
        train_loss_cum += loss * num_samples_batch
        acc_cum += accuracy(logits, y_batch) * num_samples_batch

    # average the accumulated statistics
    avg_train_loss = train_loss_cum / num_samples_epoch
    avg_acc = acc_cum / num_samples_epoch
    test_acc = evaluate(model)
    epoch_duration = time.time() - t

    # print some infos
    print(f'Epoch {epoch} | Train loss: {train_loss_cum.item():.4f} | '
          f' Train accuracy: {avg_acc.item():.4f} | Test accuracy: {test_acc.item():.4f} |'
          f' Duration {epoch_duration:.2f} sec')

    # save checkpoint of model
    if epoch % 5 == 0 and epoch > 0:
        save_path = f'model_epoch_{epoch}.pt'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()},
                   save_path)
        print(f'Saved model checkpoint to {save_path}')