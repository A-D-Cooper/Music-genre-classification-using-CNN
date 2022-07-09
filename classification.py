from classifier import CNN_classifier
from torch import nn
from torch.optim import optimizer as optim
import torch
import numpy as np
import os
import sys
from PIL import Image
from math import floor


def train(model, train_data, valid_data, batch_size=36, weight_decay=0.0, learning_rate=0.0001, num_epochs=15):
    checkpoint_path = "checkpoints/"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # train_loader1 = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

    # model.to('cuda')

    iters, losses = [], []
    iters_sub, train_accs, val_accs = [], [], []

    for i in range(1, num_epochs + 1):

        model.train()
        # x, y = train_data
        train_loader1 = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for x, y in train_loader1:
            # x = x.to('cuda')
            # y = y.to('cuda')

            pred = model.forward(x)

            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print(pred, y)

        iters.append(i)
        # losses.append(float(loss)/batch_size)
        losses.append(float(loss))

        iters_sub.append(i)
        # train_cost = float(loss.cpu().detach().numpy())
        train_cost = float(loss.detach().numpy())
        train_acc = get_accuracy(model, train_data)
        train_accs.append(train_acc)
        val_acc = get_accuracy(model, valid_data)
        val_accs.append(val_acc)
        print("Epoch %d. [Val Acc %.0f%%] [Train Acc %.0f%%, Loss %f]" % (i, val_acc * 100, train_acc * 100, train_cost))

        if (checkpoint_path is not None):  # and i > 10:
            torch.save(model.state_dict(), checkpoint_path.format(i))

    return iters, losses, iters_sub, train_accs, val_accs


def get_accuracy(model, data):
    # note: why should we use a larger batch size here?
    loader = torch.utils.data.DataLoader(data, batch_size=256)

    model.eval()  # annotate model for evaluation (why do we need to do this?)
    # to tell the model we want evaluation and not training

    correct = 0
    total = 0
    for imgs, labels in loader:
        # imgs = imgs.to('cuda')
        y_hat = model.forward(imgs)

        # pred = y_hat.detach().cpu().numpy()
        pred = y_hat.detach().numpy()

        pred = np.argmax(pred, axis=1)
        # print(pred)

        # labels = labels.cpu().detach().numpy()
        labels = labels.detach().numpy()

        # print(labels)
        correct += np.sum(pred == labels)
        total += labels.shape[0]

def data_loader(path, val_split=0.1, test_split=0.2):

    data = [[],[]]
    classes = {}
    labels = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for s in files:
            name, genre = s.split('-')
            spec = Image.open(s)
            data[0].append(spec)
            try:
                label = classes[genre]
            except KeyError:
                classes[genre] = labels
                label = labels
                labels += 1
            data[1].append(label)

    data_points = len(data[1])
    test_points = floor(data_points * test_split)
    val_points = floor(data_points * val_split)
    train_points = data_points - test_points - val_points

    train_data = [data[0][0:train_points], data[1][0:train_points]]
    val_data = [data[0][train_points: train_points + val_points], data[1][train_points: train_points + val_points]]
    test_data = [data[0][train_points + val_points:], data[1][train_points + val_points:]]

    return train_data, val_data, test_data

if __name__ == "__main__":
    classifier = CNN_classifier()
