import os
import shutil
import argparse
from pathlib import Path
from time import time
from collections import defaultdict
import torch
import numpy as np
import pandas as pd


torch.manual_seed(0)

def allocate(a):
    a[a<0] = 0
    if a.sum() <= 1:
        return a
    else:
        return a/a.sum()

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
    return (
        loss.item(),
        ret(output, y)
    )

def test_batch(X, y):
    X = torch.Tensor(X)
    output = net(X)
    output = output.detach().numpy()
    return (
        output,
        ret(output, y)
    )

def train():
    print('Train...')
    start_time = time()
    net.reset_parameters() # repeat training in jupyter notebook
    summary = []
    # loop over epoch and batch
    for e in range(epoch):
        current_epoch = defaultdict(list)
        for i, X, target, y in data.train():
            tr_loss, tr_ret = train_batch(X, target, y)
            current_epoch['tr_loss'].append(tr_loss)
            current_epoch['tr_ret'].extend(tr_ret)

        # evaluate
        for i, X, y in data.valid():
            _, va_ret = test_batch(X, y)
            current_epoch['va_ret'].extend(va_ret)

        # loss, train average daily % return, valid ...
        aggregate = [np.mean(current_epoch['tr_loss']),
                     np.mean(current_epoch['tr_ret'])*100,
                     np.mean(current_epoch['va_ret'])*100]
        print("epoch:{:3d}, tr_loss:{:+.3f}, tr_ret:{:+.3f}, va_ret:{:+.3f}".format(
            e+1, *aggregate))
        # only save the best model on validation set
        if not summary or aggregate[-1] > summary[-1][-1]:
            best = e+1
            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'criterion': criterion.state_dict()
            }, save_dir.joinpath('state.pt'))
        summary.append(aggregate)

    summary = pd.DataFrame(summary, columns=['tr_loss', 'tr_ret', 'va_ret']).to_csv(
        save_dir.joinpath('train_summary.csv'))
    pd.DataFrame(current_epoch['tr_ret'], columns=['tr_ret']).to_csv(
        save_dir.joinpath('train_last_epoch.csv'))
    print('Training finished after {:.1f}s'.format(time()-start_time))
    print('Best epoch: {}'.format(best))
    print('-'*20)


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
    outputs = []

    for i, X, y in data.test():
        output, r = test_batch(X, y)
        outputs.extend(zip(i, output))
        summary.extend(zip(i, r))

        if online_train:
            net.train()
            current_epoch = []
            for j, X, target, y in data.online_train(online_train_batch_num, p):
                current_epoch.append(train_batch(X, target, y))
            net.eval()
            current_epoch = np.array(current_epoch)
    summary = pd.DataFrame(summary, columns=['index', 'ret'])
    summary = summary.set_index('index')
    summary.to_csv(save_dir.joinpath('test_summary.csv'))
    print('ret: {:+.3f}'.format((summary['ret']+1).prod()))

    outputs = dict(outputs)
    outputs = pd.DataFrame(outputs).T
    outputs.to_csv(save_dir.joinpath('test_output.csv'))
    print((outputs.sum(axis=0)/outputs.values.sum()).round(3))


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
    os.environ['CONFIG_LOCAL_DIR'] = args.path
    # variables defined here are global/model level
    save_dir = Path(args.path)
    from config_global import epoch, net, optimizer, criterion, data, online_train
    if not args.test:
        train()
    net, optimizer, criterion = load_model(save_dir.joinpath('state.pt'))
    test()

