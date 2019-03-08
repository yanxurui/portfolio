import inspect
import torch.optim as optim
from model import CNN, ReturnAsLoss
from dataset import StockData
from networks import *


data_path = 'data.csv'
features = ['Open', 'High', 'Low', 'Close']
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'INTC', 'CSCO', 'CMCSA', 'PEP', 'NFLX', 'ADBE']

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

net=SeedLSTM((train_batch_size, len(features), len(stocks)+1, window))

