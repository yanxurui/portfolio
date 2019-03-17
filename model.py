import torch
import torch.nn as nn
import torch.nn.functional as F

# build network
class Base(nn.Module):
    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()


class CNN(Base):
    def __init__(self, shape):
        super(CNN, self).__init__()
        x = torch.zeros(shape) # dummy inputs used to infer shapes of layers
        # print(x.shape)

        self.conv1 = nn.Conv2d(x.shape[1], 128, (1, 3), bias=False)
        x = self.conv1(x)
        # print(x.shape)

        self.conv2 = nn.Conv2d(x.shape[1], 64, (1, 3), bias=False)
        x = self.conv2(x)
        # print(x.shape)

        self.conv3 = nn.Conv2d(x.shape[1], 32, (1, x.shape[3]), bias=False)
        x = self.conv3(x)
        # print(x.shape)

        self.conv4 = nn.Conv2d(x.shape[1], 1, (1, 1), bias=False)
        x = self.conv4(x)
        # print(x.shape)

        x = x.view(x.shape[0], -1) # 4d -> 2d
        # print(x.shape)

        self.output = nn.Softmax(dim=1)
        x = self.output(x)
        # print(x.shape)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.output(x)
        return x


class Softmax_T(nn.Module):
    def __init__(self, dim=None, T=25):
        super().__init__()
        self.dim = dim
        self.T = T

    def forward(self, input):
        if self.training:
            return F.softmax(input, dim=self.dim)
        else:
            return F.softmax(input/self.T, dim=self.dim)


class CNN_T(CNN):
    '''with temperature to calibrate confidence
    '''
    def __init__(self, shape, **kargs):
        super().__init__(shape)
        self.output = Softmax_T(dim=1, **kargs)


class CNN_Tanh(CNN):
    '''replace softmax with tanh
    '''
    def __init__(self, shape):
        super().__init__(shape)
        self.output = nn.Tanh()


class CNN_Sigmoid(CNN):
    '''replace softmax with sigmoid and then do x/x.sum()
    '''
    def __init__(self, shape):
        super().__init__(shape)
        self.output = nn.Sigmoid()


class Conv2DNet(nn.Module):
    '''2d convolution across different stocks'''
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
        x = F.softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()


class RNN(Base):
    def __init__(self, shape):
        super().__init__()
        self.rnn = nn.RNN(shape[1], 20, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(20, 1, bias=False)
        self.output = Softmax_T(dim=1)

    def forward(self, x):
        _x = x.permute(0, 2, 3, 1) # (batch, asset, sequence, feature)
        _x = [_x[:, i] for i in range(_x.shape[1])]
        output = []
        for a in _x:
            o, h = self.rnn(a)
            output.append(self.fc(h[-1])) # the last hidden state of last layer
        output = torch.stack(output).squeeze(dim=-1).permute(1, 0)
        return self.output(output)


class LSTM(Base):
    def __init__(self, shape):
        super().__init__()
        self.lstm = nn.LSTM(shape[1], 20, num_layers=2, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(20, 1, bias=False)
        self.output = Softmax_T(dim=1)

    def forward(self, x):
        _x = x.permute(0, 2, 3, 1) # (batch, asset, sequence, feature)
        _x = [_x[:, i] for i in range(_x.shape[1])]
        output = []
        for a in _x:
            o, (h, c) = self.lstm(a)
            output.append(self.fc(h[-1])) # the last hidden state of last layer
        output = torch.stack(output).squeeze(dim=-1).permute(1, 0)
        return self.output(output)


class OnehotLSTM(nn.Module):
    '''seed lstm structure, add onehot to represent assets
    '''
    def __init__(self, shape):
        super(OnehotLSTM, self).__init__()
        self.lstm = nn.LSTM(shape[1]+ shape[2], 20)
        self.projection = nn.Linear(20, 1, bias=False)

    def onehot(self, x, i, length):
        _x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], length)], 2)
        _x[:,:, i+x.shape[2]] = 1.
        return _x

    def forward(self, x):
        x_break = [x[:,:,i,:] for i in range(x.shape[2])]
        x_lstmed = []
        for i in range(len(x_break)):
            _x = x_break[i]
            o, (h, c) = self.lstm(self.onehot(_x.permute(2, 0, 1), i, len(x_break)))
            x_lstmed.append(h)

        x_dim_align = torch.stack(x_lstmed).squeeze()
        if len(x_dim_align.shape) < 2:
            x_dim_align = x_dim_align.unsqueeze(1)
        assert len(x_dim_align.shape) == 2

        x_permuted = x_dim_align.permute(1, 0)

        return F.softmax(x_permuted, dim=1)

    def reset_parameters(self):
        pass

class CrossRNN(nn.Module):
    '''RNN with all asset price considered, i.e. input size 44'''
    def __init__(self, shape):
        super(RnnNet, self).__init__()

        x = torch.zeros(shape) # dummy inputs used to infer shapes of layers
        self.rnn1 = nn.RNN(x.shape[1] * x.shape[2], x.shape[2], bias=False, dropout=0.6, num_layers=2)
        h0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        x = x.permute(3, 0, 1, 2)

        x = x.view(x.shape[0], x.shape[1], -1)
        x, h = self.rnn1(x, h0)
        x = x.permute(1, 0, 2)
        self.conv = nn.Conv1d(x.shape[1], 1, 1, bias=False)
        x = self.conv(x)

    def forward(self, x):

        h0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)

        x, h = self.rnn1(x, h0)
        x = x.permute(1, 0, 2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return F.softmax(x, dim=-1)

    def reset_parameters(self):
        pass

class CrossLSTM(nn.Module):
    '''LSTM with all asset price considered, i.e. input size 44'''
    def __init__(self, shape):
        super(CrossLSTM, self).__init__()
        x = torch.zeros(shape) # dummy inputs used to infer shapes of layers
        self.rnn1 = nn.LSTM(x.shape[1] * x.shape[2], x.shape[2], bias=False, dropout=0.6, num_layers=2)
        h0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        c0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        x = x.permute(3, 0, 1, 2)

        x = x.view(x.shape[0], x.shape[1], -1)
        x, h = self.rnn1(x, (h0, c0))
        x = x.permute(1, 0, 2)
        self.conv = nn.Conv1d(x.shape[1], 1, 1, bias=False)
        x = self.conv(x)

    def forward(self, x):

        h0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        c0 = nn.Parameter(torch.randn((2 * 1, x.shape[0], x.shape[2])), requires_grad=True)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)

        x, h = self.rnn1(x, (h0, c0))
        x = x.permute(1, 0, 2)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return F.softmax(x, dim=-1)

    def reset_parameters(self):
        pass



# define loss function
class ReturnAsLoss(nn.Module):
    def __init__(self):
        super(ReturnAsLoss, self).__init__()

    def forward(self, output, y):
        '''negative logarithm return'''
        return -torch.sum(torch.log(torch.sum(output * (y+1), dim=1)))


class CustomizedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, y):
        return -torch.mean(torch.sum(output * y, dim=1))


class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        self._criteria = nn.CrossEntropyLoss()

    def forward(self, output, y):
        y_copy = y.clone()
        y_copy[:, 0] += 0.005 # be conservative
        return self._criteria(output, y_copy.argmax(dim=1))


class Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self._criteria = nn.BCELoss()

    def forward(self, output, y):
        y_copy = y.clone()
        y_copy[y>0] = 0.9 # increase precision for positive class
        y_copy[y<0] = 0
        return self._criteria(output, y_copy)


class PenalizeSingle(nn.Module):
    '''Add penalize to oracle error function'''
    def __init__(self):
        super().__init__()
        self._criteria = nn.CrossEntropyLoss()

    def penalize(self, x):
        return torch.dot(x, x)

    def forward(self, output, y):
        y_copy = y.clone()
        y_copy[:, 0] += 0.005 # be conservative
        return self._criteria(output, y_copy.argmax(dim=1)) + self.penalize(y_copy)
