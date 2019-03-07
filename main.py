import os
import shutil
import argparse
from pathlib import Path
from time import time

import torch
import numpy as np
import pandas as pd

torch.manual_seed(0)

def allocate(a):
    a[a<0] = 0
    if a.sum() == 0:
        a[0] = 1
    else:
        a = a/a.sum()
    return a

def ret(output, y):
    output = np.apply_along_axis(allocate, -1, output)
    return (output*y).sum(axis=1)

def train_batch(X, target, y, net, optimizer, criterion):
    X, target = torch.Tensor(X), torch.Tensor(target)
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(X)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
    output = output.detach().numpy()
#    print('weights in batch', output)
    return (
        loss.item(),
        np.mean(output.argmax(1) == y.argmax(1)), # accuracy
        ret(output, y).prod() # cumulative return in this batch
    )

def train():
    print('Train...')
    start_time = time()
    net.reset_parameters() # repeat training in jupyter notebook
    summary = []
    columns=['loss', 'acc', 'ret']

    # loop over epoch and batch
    for e in range(epoch):
        current_epoch = []
        for i, X, target, y in data.train():
            current_epoch.append(train_batch(X, target, y, net, optimizer, criterion))
        current_epoch = np.array(current_epoch)
        summary.append(current_epoch.mean(axis=0))
        print("epoch:{:3d}, loss:{:+.3f}, acc:{:.3f}, ret:{:+.3f}".format(
            e+1,
            *(summary[-1])))
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict()
        }, save_dir.joinpath('state.pt'))
    pd.DataFrame(summary, columns=columns).to_csv(save_dir.joinpath('train_summary.csv'))
    pd.DataFrame(current_epoch, columns=columns).to_csv(save_dir.joinpath('train_last_epoch.csv'))
    print('Training finished after {:.1f}s'.format(time()-start_time))
    print('*'*20)


def load_model(path):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion.load_state_dict(checkpoint['criterion'])
    return net, optimizer, criterion

def reallocate(w):
    # assets that have weights lower than cash are eliminated
    cash = w[:, 0][:, None]
    w[w < cash] = 0
    w[:, 0] += 1-w.sum(axis=1)
    return w

def test():
    # always load model from disk 
    #   1. to repeat test without training
    #   2. for the sake of online learning
    print('Test...')
    model = load_model(save_dir.joinpath('state.pt'))
    net = model[0]
    summary = []
    weights = []

    for i, X, y in data.test():
        X = torch.Tensor(X)
        output = net(X)
        output = output.detach().numpy()
        weights.append(output)
        acc = output.argmax(1) == y.argmax(1)

        summary.extend(zip(i, acc, ret(output, y)))
        if online_train:
            net.train()
            current_epoch = []
            for j, X, target, y in data.online_train(online_train_batch_num, p):
                current_epoch.append(train_batch(X, target, y, *model))
            net.eval()
            current_epoch = np.array(current_epoch)
    summary = pd.DataFrame(summary, columns=['index', 'acc', 'ret'])
    summary = summary.set_index('index')
    summary.to_csv(save_dir.joinpath('test_summary.csv'))
    print('acc: {:.3f} ret: {:+.3f}'.format(
        summary['acc'].mean(),
        summary['ret'].prod()
        ))
    weights = np.array(weights)
    print(weights.sum(axis=0)/weights.sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Null')
    parser.add_argument('path', help='path of experiment, must contain config.py')
    parser.add_argument('--test', action='store_true', help='test only')
    args = parser.parse_args()
    # variables defined here are global
    save_dir = Path(args.path)
    shutil.copy(save_dir.joinpath('config.py'), './') # copy and overwrite
    from config_global import *

    if not args.test:
        train()
    test()

