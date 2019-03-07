import os
import sys
import inspect
import torch.optim as optim
from model import CNN, ReturnAsLoss
from dataset import StockData


data_path = 'data.csv'
features = ['Open', 'High', 'Low', 'Close']
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'INTC', 'CSCO', 'CMCSA', 'PEP', 'NFLX', 'ADBE']

# train
epoch = 100 # repeat times
train_batch_num = 100
train_batch_size = 10
window = 5 # using historical price in the past few days as features
learning_rate = 0.001

# test
test_batch_num = 200
test_batch_size = 1
online_train = False
online_train_batch_num = 10
p = 0.01

# construct model
net = CNN((train_batch_size, len(features), len(stocks)+1, window))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = ReturnAsLoss()
data = StockData(data_path, features=features, stocks=stocks,
    train_batch_num=train_batch_num,
    train_batch_size=train_batch_size,
    test_batch_num=test_batch_num,
    test_batch_size=test_batch_size,
    window=window)

######### LOAD LOCAL CONFIG, OVERWRITE DEFAULT CONFIG ############
with open(os.path.join(sys.argv[1], 'config.py'), 'r') as f:
    exec(f.read())
##################################################################


# print the current config
# at module level, globals and locals are the same dictionary
_namespace = globals().copy()
for k,v in _namespace.items():
    if not (inspect.isbuiltin(v) or inspect.isclass(v) or inspect.ismodule(v) or k[0] == '_'):
        print("{}: {}".format(k,v))
