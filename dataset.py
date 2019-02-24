import logging
import numpy as np
import pandas as pd
import torch
idx = pd.IndexSlice

class Dataset:
    '''
    input:
        data is 3d: (features, assets, window)
    return a batch:
        X: is a 4d array (batch_size, OHLC, assets, window)
           All price is divided by the closing price in the first day and subtracted by 1
        y: is a binary 2d array (batch_size, assets)
    '''
    def __init__(self, data, features=None, stocks=None,
                       train_batch_num=200,
                       batch_size=10,
                       window=10):
        if type(data) is str: # path is given
            data = self._load_data(data, features, stocks)
        assert len(data.shape) == 3
        self.train_end = train_batch_num * batch_size + window
        self.n = data.shape[-1]
        if self.train_end > self.n:
            raise ValueError("There is not enough data. Try to use a smaller train_batch_num")
        # logger = logging.getLogger('nn')
        # logger.info("Training from {} to {}".format(data[:, :, window].name, data[:, :, self.train_end-1].name))
        # logger.info("Testing from {} to {}".format(data[:, :, self.train_end].name, data[:, :, -1].name))
        shape = data.shape
        cash = np.ones((shape[0], 1, shape[-1]))
        data = np.concatenate((cash, data), axis=1)
        self.data = data
        self.train_batch_num = train_batch_num
        self.batch_size = batch_size
        self.features = shape[0]
        self.window = window
    
    def _load_data(self, data_path, features, stocks):
        if features is None:
            features = ['Open', 'High', 'Low', 'Close']
        index = idx[features, :]
        if stocks is not None:
            index = idx[features, stocks]
        data_pd = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
        data_pd_truncated = data_pd.T.loc[index, idx[:]]
        return data_pd_truncated.values.reshape((len(features), -1, len(data_pd)))

    def _get_batch(self, begin, end):
        assert begin - self.window >= 0
        assert end <= self.n
        batch_size = end - begin
        X = []
        y = []
        # import pdb
        # pdb.set_trace()
        for b in range(begin, end):
            # normalized by the closing price in the first day, using broadcast
            X.append(self.data[:,:,b-self.window:b]/(self.data[-1,:,b-self.window][None,:,None])-1)
        X = np.array(X) # 4d
        # X = np.concatenate((torch.zeros((batch_size, self.features, 1, self.window)), X), axis=2)
        # price movement relative to the previous day
        tmp = self.data[-1,:,begin-1:end] # 3d->2d
        y = np.diff(tmp, axis=-1)/tmp[:,:-1] + 1
        y = np.transpose(y)
        # y = np.concatenate((torch.ones((batch_size, 1)), y), axis=1)
        return X, y

    def train_batch(self):
        batch_offset = self.window # history is the previous M days, batch_offset is the next trading day
        for _ in range(self.train_batch_num):
            yield self._get_batch(batch_offset, batch_offset+self.batch_size)
            batch_offset += self.batch_size

    def test_batch(self, batch_size=None):
        '''the rest period for test
        '''
        if batch_size is None:
            batch_size = self.batch_size
        batch_offset = self.train_end
        while batch_offset + batch_size <= self.n:
            yield self._get_batch(batch_offset, batch_offset+batch_size)
            batch_offset += batch_size

    def baseline(self):
        '''average closing price of out of sample period
        '''
        out_sample = self.data[-1, :, self.train_end:]/self.data[-1, :, self.train_end-1][:, None]
        return out_sample.mean(axis=0)


if __name__ == '__main__':
    from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal
    data = np.arange(1,2*2*20+1).reshape(2, 2, 20)
    d = Dataset(data, train_batch_num=2, batch_size=3, window=2)
    # _get_batch
    X, y = d._get_batch(2, 2+3)
    assert_equal(X.shape, (3, 2, 2+1, 2)) # (batch_size, features, assets, window)
    assert_array_equal(X[0, 0, 0], [0, 0]) # cash
    assert_equal(X[0, 0, 1, 1], data[0, 1-1, 1]/data[-1, 0, 0]-1)
    assert_equal(X[1, 1, 2, 1], data[1, 2-1, 2]/data[-1, 1, 1]-1)

    assert_equal(y.shape, (3, 3)) # (batch_size, assets)
    assert_equal(y[0, 0], 1) # cash
    assert_equal(y[0, 1], data[-1, 0, 2]/data[-1, 0, 1])

    # train_batch
    g = d.train_batch()
    X, y = next(g)
    X, y = next(g) # second batch 5 6 7
    assert_array_equal(X[0, 0, 0], [0, 0]) # cash
    assert_equal(X[0, 0, 1, 1], data[0, 0, 4]/data[-1, 0, 3]-1)
    assert_equal(X[1, 0, 1, 1], data[0, 0, 5]/data[-1, 0, 4]-1)

    # batch_test
    g = d.test_batch()
    X, y = next(g) # first batch 8 9 10
    assert_equal(X[1, 0, 1, 1], data[0, 0, 8]/data[-1, 0, 7]-1)
