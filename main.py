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

flatten = lambda l: [item for sublist in l for item in sublist]
daily_ret = lambda ret_list, days: 100*(pow(np.prod(ret_list), 1/days)-1)

def ret(output, y):
    output = np.apply_along_axis(allocate, -1, output)
    return (output*y).sum(axis=1)

def train_batch(X, target, y):
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

def test_batch(X, y):
    X = torch.Tensor(X)
    output = net(X)
    output = output.detach().numpy()
    return (
        output,
        output.argmax(1) == y.argmax(1),
        ret(output, y)
    )

def train():
    print('Train...')
    start_time = time()
    net.reset_parameters() # repeat training in jupyter notebook
    summary = []
    columns=['tr_loss', 'tr_acc', 'tr_ret', 'va_acc', 'va_ret']

    # loop over epoch and batch
    for e in range(epoch):
        train_epoch = []
        for i, X, target, y in data.train():
            train_epoch.append(train_batch(X, target, y))
        train_epoch = np.array(train_epoch)

        # evaluate
        valid_epoch = []        
        for i, X, y in data.valid():
            _, acc, r = test_batch(X, y)
            valid_epoch.extend(zip(acc, r))
        valid_epoch = np.array(valid_epoch)

        current_epoch = [
            # loss
            train_epoch[:,0].mean(),
            # train acc
            train_epoch[:,1].mean(),
            # train average daily % return
            daily_ret(train_epoch[:,2], data.train_batch_num*data.train_batch_size),
            # valid acc
            valid_epoch[:,0].mean(),
            # valid average daily % return
            daily_ret(valid_epoch[:,1], data.valid_batch_num*data.valid_batch_size),
        ]

        print("epoch:{:3d}, {}:{:+.3f}, {}:{:.3f}, {}:{:+.3f}, {}:{:.3f}, {}:{:+.3f}".format(
            e+1,
            *flatten(zip(columns, current_epoch)))
        )

        # only save the best model on validation set
        if not summary or current_epoch[-1] > summary[-1][-1]:
            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'criterion': criterion.state_dict()
            }, save_dir.joinpath('state.pt'))
        summary.append(current_epoch)

    pd.DataFrame(summary, columns=columns).to_csv(save_dir.joinpath('train_summary.csv'))
    pd.DataFrame(train_epoch, columns=columns[:3]).to_csv(save_dir.joinpath('train_last_epoch.csv'))
    print('Training finished after {:.1f}s'.format(time()-start_time))
    print('*'*20)


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
    summary = []
    outputs = {}

    for i, X, y in data.test():
        output, acc, r = test_batch(X, y)
        outputs.update(dict(zip(i, output)))
        summary.extend(zip(i, acc, r))

        if online_train:
            net.train()
            current_epoch = []
            for j, X, target, y in data.online_train(online_train_batch_num, p):
                current_epoch.append(train_batch(X, target, y))
            net.eval()
            current_epoch = np.array(current_epoch)
    summary = pd.DataFrame(summary, columns=['index', 'acc', 'ret'])
    summary = summary.set_index('index')
    summary.to_csv(save_dir.joinpath('test_summary.csv'))
    print('acc: {:.3f} ret: {:+.3f}'.format(
        summary['acc'].mean(),
        summary['ret'].prod()
        ))

    outputs = pd.DataFrame(outputs)
    outputs = outputs.T
    outputs.to_csv(save_dir.joinpath('test_output.csv'))
    print(outputs.sum(axis=0)/outputs.values.sum())


def load_model(path):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion.load_state_dict(checkpoint['criterion'])
    return net, optimizer, criterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Null')
    parser.add_argument('path', help='path of experiment, must contain config.py')
    parser.add_argument('--test', action='store_true', help='test only')
    args = parser.parse_args()
    # variables defined here are global
    save_dir = Path(args.path)
    from config_global import epoch, net, optimizer, criterion, data
    if not args.test:
        train()
    net, optimizer, criterion = load_model(save_dir.joinpath('state.pt'))
    test()
