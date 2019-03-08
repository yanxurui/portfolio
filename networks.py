import torch
import torch.nn as nn
import torch.nn.functional as F


# build network
class Conv2DNet(nn.Module):
    '''convolution across different stocks'''
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

class SeedLSTM(nn.Module):
    def __init__(self, shape):
        super(SeedLSTM, self).__init__()
        self.lstm = nn.LSTM(shape[1], 20)
        self.projection = nn.Linear(20, 1, bias=False)


    def forward(self, x):
        x_break = [x[:,:,i,:] for i in range(x.shape[2])]
        x_lstmed = []
        for i in range(len(x_break)):
            _x = x_break[i]
            o, (h, c) = self.lstm(_x.permute(2, 0, 1))
            x_lstmed.append(torch.tanh(self.projection(h)))

        x_dim_align = torch.stack(x_lstmed).squeeze()
        if len(x_dim_align.shape) < 2:
            x_dim_align = x_dim_align.unsqueeze(1)
        assert len(x_dim_align.shape) == 2

        x_permuted = x_dim_align.permute(1, 0)

        return F.softmax(x_permuted, dim=1)

    def reset_parameters(self):
        pass



class OnehotLSTM(nn.Module):
    def __init__(self, shape):
        super(OnehotLSTM, self).__init__()
        self.lstm = nn.LSTM(shape[1]+ shape[2], 1)

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
