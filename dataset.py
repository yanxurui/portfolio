import numpy as np
import pandas as pd
idx = pd.IndexSlice


def count(start=0, step=1, size=None):
    # count(10) --> 10 11 12 13 14 ...
    # count(2.5, 0.5) -> 2.5 3.0 3.5 ...
    # count(1, 2, 3) -> 1, 3, 5
    n = start
    i = 0
    while True:
        if size is not None and i >=size:
            return
        yield n
        n += step
        i += 1


def discounted(a, gamma=0.3):
    g = 0
    for e in a[::-1]:
        g = e + gamma*g
    return g


class StockData:
    '''
    input:
        data is 3d: (features, assets, window)
    return a batch:
        X: is a 4d array (batch_size, OHLC, assets, window)
           All price is divided by the closing price in the first day and then subtracted by 1
        y: is a binary 2d array (batch_size, assets)
    '''
    def __init__(self, path, window=10, features=None, stocks=None,
                       train_batch_num=200, train_batch_size=10,
                       valid_batch_num=1, valid_batch_size=100,
                       test_batch_num=None, test_batch_size=1):
        data = self._load_data(path, features, stocks)
        shape = data.shape
        assert len(shape) == 3
        self.n = shape[-1] # how many training examples
        self.train_end = window + train_batch_num * train_batch_size
        self.valid_end = self.train_end + valid_batch_num * valid_batch_size
        self.test_end = (self.valid_end + test_batch_num * test_batch_size) if test_batch_num else -1
        if self.valid_end > self.n:
            raise ValueError("There is not enough data. Try to use a smaller train_batch_num")
        idx_to_date = lambda i: self.data_raw.index[i].strftime('%Y-%m-%d')
        print("Training   from {} to {}".format(idx_to_date(window), idx_to_date(self.train_end-1)))
        print("Validation from {} to {}".format(idx_to_date(self.train_end), idx_to_date(self.valid_end-1)))
        print("Test       from {} to {}".format(idx_to_date(self.valid_end), idx_to_date(self.test_end)))
        cash = np.ones((shape[0], 1, shape[-1]))
        self.data = np.concatenate((cash, data), axis=1)
        self.features = shape[0]
        self.window = window
        self.train_batch_num = train_batch_num
        self.train_batch_size = train_batch_size
        self.valid_batch_num = valid_batch_num
        self.valid_batch_size = valid_batch_size
        self.test_batch_num = test_batch_num
        self.test_batch_size = test_batch_size


    def _load_data(self, path, features, stocks):
        if features is None:
            features = ['Open', 'High', 'Low', 'Close']
        index = idx[features, :]
        if stocks is not None:
            index = idx[features, stocks]
        self.data_raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        data_pd_truncated = self.data_raw.T.loc[index, idx[:]]
        return data_pd_truncated.values.reshape((len(features), -1, len(self.data_raw)))

    def _historical_price(self, begin, end):
        # return historical price of an example
        # normalized by the closing price in the first day
        return self.data[:,:,begin:end]/(self.data[-1,:,begin][None,:,None])-1

    def _price_change(self, begin, end):
        # price movement relative to the previous day in a batch
        tmp = self.data[-1,:,begin-1:end] # 3d->2d
        y = np.diff(tmp, axis=-1)/tmp[:,:-1] + 1
        return np.transpose(y)

    def _get_batch(self, begin, end):
        assert begin - self.window >= 0
        assert end <= self.n
        batch_size = end - begin
        X = []
        y = []
        indices = list(range(begin, end))
        for b in indices:
            X.append(self._historical_price(b-self.window, b))
        X = np.array(X) # 4d
        y = self._price_change(begin, end)
        return indices, X, y

    def _get_train_batch(self, begin, end):
        indices, X, y = self._get_batch(begin, end)
        target = y
        return indices, X, target, y

    def _get_test_batch(self, begin, end):
        return self._get_batch(begin, end)

    def train(self):
        # history is the previous M days, offset is the current trading day
        for offset in count(start=self.window, step=self.train_batch_size, size=self.train_batch_num):
            yield self._get_train_batch(offset, offset+self.train_batch_size)

    def valid(self):
        for offset in count(start=self.train_end, step=self.valid_batch_size, size=self.valid_batch_num):
            yield self._get_test_batch(offset, offset+self.valid_batch_size)

    def test(self):
        '''the rest period for test
        '''
        for offset in count(start=self.valid_end, step=self.test_batch_size, size=self.test_batch_num):
            if offset + self.test_batch_size > self.n:
                return
            self.test_end = offset+self.test_batch_size # save test_end for online training
            yield self._get_test_batch(offset, self.test_end)

    def online_train(self, batch_num=10, p=0.2):
        # sample by geometric distribution
        offsets = self.test_end - self.train_batch_size - (np.random.geometric(p=p, size=batch_num) - 1)
        offsets = np.clip(offsets, a_min=self.window, a_max=None)
        for offset in offsets:
            yield self._get_train_batch(offset, offset+self.train_batch_size)

    def market(self, begin, end=-1):
        market_average = self.data_raw.iloc[begin:end]['Close']/self.data_raw.iloc[begin-1]['Close']
        return market_average.mean(axis=1)


class StockData_CR(StockData):
    # cumulative return in the following days as criteria
    def _get_train_batch(self, begin, end):
        indices, X, target, y = super()._get_train_batch(begin, end)

        forward_days = 3
        t = self._price_change(begin, end+forward_days-1) # (batch_size+3, assets)
        # split into overlapping subarrays
        t = [t[i:i+forward_days] for i in range(0, len(t)-forward_days+1)]
        t = np.array(t)
        t = t.prod(axis=1)
        return indices, X, t, y


class StockData_DR(StockData):
    # discounted return
    def _get_train_batch(self, begin, end):

        indices, X, target, y = super()._get_train_batch(begin, end)

        forward_days = 3
        t = self._price_change(begin, end+forward_days-1) # (batch_size+3, assets)
        t = t - 1
        # split into overlapping subarrays
        t = [t[i:i+forward_days] for i in range(0, len(t)-forward_days+1)]
        t = np.array(t)
        t = np.apply_along_axis(discounted, 1, t)
        return indices, X, t, y



if __name__ == '__main__':
    from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal
    # count
    c = count(1)
    assert_equal([next(c) for _ in range(3)], [1,2,3])
    c = count(2.5, 0.5)
    assert_equal([next(c) for _ in range(3)], [2.5,3,3.5])
    c = count(1, 2, 3)
    # The use of the list() instead of [] suppress the StopIteration exception thrown by next()
    assert_equal(list(next(c) for _ in range(4)), [1,3,5])
    c = count(0, 1, 0)
    assert_equal(list(next(c) for _ in range(1)), [])

    # discounted
    assert_equal(discounted([0.1, 0.2], gamma=0.5), 0.2)


    # patch
    data = np.arange(1,2*2*20+1).reshape(2, 2, 20)
    def _load_data_dec(func, data):
        def wrapper(self, *args):
            func(self, *args)
            return data
        return wrapper
    StockData._load_data = _load_data_dec(StockData._load_data, data)
    d = StockData('data.csv', window=2,
        train_batch_num=2, train_batch_size=3,
        valid_batch_num=0, 
        test_batch_size=3)

    # _get_batch
    i, X, y = d._get_batch(2, 2+3)
    assert_equal(i, [2,3,4])
    assert_equal(X.shape, (3, 2, 2+1, 2)) # (batch_size, features, assets, window)
    assert_array_equal(X[0, 0, 0], [0, 0]) # cash
    assert_equal(X[0, 0, 1, 1], data[0, 1-1, 1]/data[-1, 0, 0]-1)
    assert_equal(X[1, 1, 2, 1], data[1, 2-1, 2]/data[-1, 1, 1]-1)

    assert_equal(y.shape, (3, 3)) # (batch_size, assets)
    assert_equal(y[0, 0], 1) # cash
    assert_equal(y[0, 1], data[-1, 0, 2]/data[-1, 0, 1])

    # train
    g = d.train()
    i, X, t, y = next(g)
    i, X, t, y = next(g) # second batch 5 6 7
    assert_equal(i, [5,6,7])
    assert_array_equal(X[0, 0, 0], [0, 0]) # cash
    assert_equal(X[0, 0, 1, 1], data[0, 0, 4]/data[-1, 0, 3]-1)
    assert_equal(X[1, 0, 1, 1], data[0, 0, 5]/data[-1, 0, 4]-1)

    # test
    g = d.test()
    i, X, y = next(g) # first batch 8 9 10
    assert_equal(i, [8,9,10])
    assert_equal(X[1, 0, 1, 1], data[0, 0, 8]/data[-1, 0, 7]-1)

    # online_train
    i2, X2, t2, y2 = next(d.online_train(batch_num=1, p=1))
    assert_array_equal(i, i2)
    assert_array_equal(X, X2)
    assert_array_equal(y, y2)

    print('Pass')

