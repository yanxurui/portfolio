import os
import sys
import traceback
import inspect
import warnings
from functools import partial
import torch.optim as optim
from model import CNN_T, CustomizedLoss
from dataset import StockData_DR


# data
data_path = 'data.csv'
features = ['Open', 'High', 'Low', 'Close']
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'INTC', 'CSCO', 'CMCSA', 'PEP', 'NFLX', 'ADBE']

# train
epoch = 100 # repeat times
window = 5 # using historical price in the past few days as features
learning_rate = 0.001
train_batch_num = 200
train_batch_size = 10
valid_batch_num = 1
valid_batch_size = 500

# test
test_batch_num = None
test_batch_size = 1
online_train = False
online_train_batch_num = 10
p = 0.01

Net = partial(CNN_T, T=25)
Optimizer = optim.Adam
Criterion = CustomizedLoss
Data = StockData_DR


######### LOAD LOCAL CONFIG, OVERWRITE DEFAULT CONFIG ############
try:
    _dir = os.environ['CONFIG_LOCAL_DIR']
    with open(os.path.join(_dir, 'config.py'), 'r') as f:
        exec(f.read()) # time consuming
except (KeyError, FileNotFoundError):
    warnings.warn("no local config file")
    traceback.print_exc()
##################################################################


# construct model
_namespace = globals().copy()
if 'data' not in _namespace:
    data = Data(data_path, features=features, stocks=stocks,
        window=window,
        train_batch_num=train_batch_num,
        train_batch_size=train_batch_size,
        valid_batch_num=valid_batch_num,
        valid_batch_size=valid_batch_size,
        test_batch_num=test_batch_num,
        test_batch_size=test_batch_size,
        online_train_batch_num=online_train_batch_num,
        p=p)
data.info()
if 'net' not in _namespace:
    net = Net((train_batch_size, len(data.features), len(data.assets), window))
if 'optimizer' not in _namespace:
    optimizer = Optimizer(net.parameters(), lr=learning_rate)
if 'criterion' not in _namespace:
    criterion = Criterion()

# print the current config
# at module level, globals and locals are the same dictionary
_namespace = globals().copy()
for k,v in _namespace.items():
    if not (inspect.isbuiltin(v) or inspect.isclass(v) or inspect.ismodule(v) or k[0] == '_'):
        print("{}: {}".format(k,v))
