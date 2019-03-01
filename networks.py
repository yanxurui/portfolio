import torch
import torch.nn as nn
import torch.nn.functional as F


# build network
class BaselineNet(nn.Module):

    def __init__(self, shape):
        super(BaselineNet, self).__init__()
        x = torch.zeros(shape) # dummy inputs used to infer shapes of layers
        print(x.shape)
        
        self.conv1 = nn.Conv2d(x.shape[1], 128, (1, 3), bias=False)
        x = self.conv1(x)
        print(x.shape)
        
        self.conv2 = nn.Conv2d(x.shape[1], 64, (1, 3), bias=False)
        x = self.conv2(x)
        print(x.shape)
        
        self.conv3 = nn.Conv2d(x.shape[1], 32, (1, x.shape[3]), bias=False)
        x = self.conv3(x)
        print(x.shape)
        
        self.conv4 = nn.Conv2d(x.shape[1], 1, (1, 1), bias=False)
        x = self.conv4(x)
        print(x.shape)
        
        x = x.view(x.shape[0], -1) # 4d -> 2d
        print(x.shape)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, dim=1)
        return x

    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()



class Conv2DNet(nn.Module):
    def __init__(self, shape):
        super(Conv2DNet, self).__init__()

        x = torch.zeros(shape) # dummy inputs used to infer shapes of layers
        print(x.shape)
        self.conv1 = nn.Conv2d(x.shape[1], 128, (3, 3), bias=False)
        x = self.conv1(x)
        print(x.shape)
        self.conv2 = nn.Conv2d(x.shape[1], 64, (3, 3), bias=False)
        x = self.conv2(x)
        print(x.shape)
        self.conv3 = nn.Conv2d(x.shape[1], 32, (3, x.shape[3]), bias=False)
        x = self.conv3(x)
        print(x.shape)
        self.conv4 = nn.Conv2d(x.shape[1], 1, (1, 1), bias=False)
        x = self.conv4(x)
        print(x.shape)
        x = x.view(-1, 5)
        print(x.shape)

        self.extract = nn.Linear(x.shape[1], 11)
        x = self.extract(x)
        print(x.shape)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 5)
        x = self.extract(x)
        #x = x.view(x.shape[0], -1)
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

class RnnNet(nn.Module):
    def __init__(self, shape):
        super(RnnNet, self).__init__()

        x = torch.zeros(shape) # dummy inputs used to infer shapes of layers
        #print(x.shape)
        self.rnn1 = nn.RNN(x.shape[1] * x.shape[2], x.shape[2], bias=False, dropout=0.6, num_layers=2)
        h0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        x = x.permute(3, 0, 1, 2)

        x = x.view(x.shape[0], x.shape[1], -1)
        #print(x.shape)
        x, h = self.rnn1(x, h0)
        x = x.permute(1, 0, 2)
        self.conv = nn.Conv1d(x.shape[1], 1, 1, bias=False)
        x = self.conv(x)
        #print(x.shape)





    def forward(self, x):

        h0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)

        #print(x.shape)
        #h0 = nn.Parameter(torch.randn((2, x.shape[1], x.shape[2])), requires_grad=True)
        x, h = self.rnn1(x, h0)
        #self.h0 = nn.Parameterh

        x = x.permute(1, 0, 2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return F.softmax(x, dim=-1)

    def reset_parameters(self):
        pass
        #for m in self.children():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #        m.reset_parameters()

# define loss function
class ReturnAsLoss(nn.Module):
    def __init__(self):
        super(ReturnAsLoss, self).__init__()

    def forward(self, output, y):
        '''negative logarithm return'''
        return -torch.sum(torch.log(torch.sum(output * y, dim=1)))

class BestStock(nn.Module):
    def __init__(self):
        super(BestStock, self).__init__()
        self._criteria = nn.CrossEntropyLoss()

    def forward(self, output, y):
        y_copy = y.clone()
        y_copy[:, 0] += 0.005 # be conservative
        return self._criteria(output, y_copy.argmax(dim=1))

#class SharpeRatio(nn.Module):
#    def __init__(self):
#        super(SharpeRatio, self).__init_()
#        self.criteria = nn.CrossEntropyLoss()
#
#    def forward(self, output, y):


