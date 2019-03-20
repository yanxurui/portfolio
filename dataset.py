import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
idx = pd.IndexSlice
np.random.seed(0)

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

    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA5', 'MA10',
       'MACD', 'MACDS', 'MACDH', 'ROC5', 'ROC10', 'EMA20', 'ATR', 'BBANDU',
       'BBANDM', 'BBANDL', 'CCI', 'RSI', 'STOCHK', 'STOCHD', 'OBV']
    stocks = ['AAPL', 'ADBE', 'AMZN', 'CMCSA', 'CSCO', 'GOOGL', 'INTC', 'MSFT',
       'NFLX', 'PEP']
    '''
    def __init__(self, path, cash=True, window=10, features=None, stocks=None,
                       train_batch_num=200, train_batch_size=10,
                       valid_batch_num=1, valid_batch_size=100,
                       test_batch_num=None, test_batch_size=1, online_train_batch_num=10, p=0.01):
        self._load_data(path, features, stocks)
        self.train_end = window + train_batch_num * train_batch_size
        self.valid_end = self.train_end + valid_batch_num * valid_batch_size
        self.test_end = (self.valid_end + test_batch_num * test_batch_size) if test_batch_num else -1
        if self.valid_end > self.n:
            raise ValueError("There is not enough data. Try to use a smaller train_batch_num")
        self.cash = cash
        self.window = window
        self.train_batch_num = train_batch_num
        self.train_batch_size = train_batch_size
        self.valid_batch_num = valid_batch_num
        self.valid_batch_size = valid_batch_size
        self.test_batch_num = test_batch_num
        self.test_batch_size = test_batch_size
        self.online_train_batch_num = online_train_batch_num
        self.p = p
        self._preprocess()

    def _idx_date(self, i):
        '''convert int index to date
        '''
        return self.data_raw.index[i].strftime('%Y-%m-%d')

    def info(self):
        print("Training   from {} to {}".format(self._idx_date(self.window), self._idx_date(self.train_end-1)))
        print("Validation from {} to {}".format(self._idx_date(self.train_end), self._idx_date(self.valid_end-1)))
        print("Test       from {} to {}".format(self._idx_date(self.valid_end), self._idx_date(self.test_end)))

    def _fi(self, *features):
        '''convert feature names to indices'''
        codes = []
        for f in features:
            c = self.features.get(f, None)
            if c is not None:
                codes.append(c)
        return codes # might be empty

    @property
    def assets(self):
        stocks = list(self.stocks) # make a copy
        if self.cash:
            stocks.insert(0, 'CASH')
        return stocks

    def _load_data(self, path, features, stocks):
        # read dataframe from csv file
        data_raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        # discard the first few raws that have nan
        for i in count(0):
            if not np.isnan(data_raw.iloc[i]).any():
                break
        self.data_raw = data_raw.iloc[i:]
        # if features or stocks is None, use all
        # idx[slice(None)] is equivalent to the idx[:]
        index = idx[features if features else slice(None), stocks if stocks else slice(None)]
        data_pd_truncated = self.data_raw.T.loc(axis=0)[index]
        features = list(data_pd_truncated.index.unique(0)) # features in order
        self.features = {f:i for i,f in enumerate(features)}
        self.stocks = list(data_pd_truncated.index.unique(1))
        # (features, stocks, time)
        self.n = len(self.data_raw) # number of days
        self.data = data_pd_truncated.values.reshape((len(features), -1, self.n))

    def _preprocess(self):
        # divide by 100
        codes = self._fi('ROC5', 'ROC10', 'RSI', 'STOCHK', 'STOCHD')
        self.data[codes,:,:] = self.data[codes,:,:]/100
        # standard scale
        codes = self._fi('CCI')
        scaler = StandardScaler()
        for c in codes:
            # X : numpy array of shape [n_samples, n_features]
            X = self.data[c,:,:].transpose()
            X = scaler.fit_transform(X)
            self.data[c,:,:] = X.transpose()

    def _normalize(self, X):
        # X is a training example
        # normalized by the closing price in the first day
        codes = self._fi('Open', 'High', 'Low', 'Close', 'Adj Close',
            'MA5', 'MA10', 'MACD', 'MACDS', 'MACDH', 'EMA20', 'ATR',
            'BBANDU', 'BBANDM', 'BBANDL')
        # in order to keep dimension, use 0:1 instead of 0. [0] doesn't work here
        X[codes,:,:] = X[codes,:,:]/X[self._fi('Close'),:,0:1] - 1
        # normalized by the value in the first day
        codes = self._fi('Volume', 'OBV')
        X[codes,:,:] = X[codes,:,:]/X[codes,:,0:1] - 1
        return X

    def _historical_period(self, begin, end):
        X = np.copy(self.data[:,:,begin:end])
        X = self._normalize(X)
        if self.cash:
            cash = np.zeros((X.shape[0], 1, X.shape[-1]))
            X = np.concatenate((cash, X), axis=1)
        return X

    def _price_change(self, begin, end):
        # closing price movement relative to the previous day in a batch
        tmp = self.data[self._fi('Close')[0],:,begin-1:end] # 3d->2d
        y = np.diff(tmp, axis=-1)/tmp[:,:-1]
        if self.cash:
            cash = np.zeros((1, y.shape[1]))
            y = np.concatenate((cash, y), axis=0)
        return np.transpose(y)

    def _get_batch(self, begin, end):
        assert begin - self.window >= 0
        assert end <= self.n
        batch_size = end - begin
        X = []
        y = []
        indices = list(range(begin, end))
        for b in indices:
            X.append(self._historical_period(b-self.window, b))
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

    def online_train(self):
        # sample by geometric distribution
        r = np.random.geometric(p=self.p, size=self.online_train_batch_num)
        offsets = self.test_end - self.train_batch_size - (r - 1)
        offsets = np.clip(offsets, a_min=self.window, a_max=None)
        for offset in offsets:
            yield self._get_train_batch(offset, offset+self.train_batch_size)

    def UBAH(self, begin, end=-1):
        '''Uniformly Buy And Hold
        a baseline representing the market movement
        '''
        close = self.data_raw.iloc[begin-1:end].loc[:,idx['Close', self.stocks]]
        ROC = close.pct_change().iloc[1:] # broadcast
        return ROC.mean(axis=1)


class StockData_CR(StockData):
    # cumulative return in the following days as criteria
    def __init__(self, *args, forward_days=3, **kargs):
        super().__init__(*args, **kargs)
        self.forward_days = forward_days

    def _get_train_batch(self, begin, end):
        indices, X, target, y = super()._get_train_batch(begin, end)

        # only consider the following day by default
        t = self._price_change(begin, end+self.forward_days-1) # (batch_size+3, assets)
        t += 1
        # split into overlapping subarrays
        t = [t[i:i+self.forward_days] for i in range(0, len(t)-self.forward_days+1)]
        t = np.array(t)
        t = t.prod(axis=1)
        return indices, X, t-1, y


class StockData_DR(StockData_CR):
    # discounted return
    def _get_train_batch(self, begin, end):

        indices, X, target, y = super()._get_train_batch(begin, end)

        t = self._price_change(begin, end+self.forward_days-1) # (batch_size+3, assets)
        # split into overlapping subarrays
        t = [t[i:i+self.forward_days] for i in range(0, len(t)-self.forward_days+1)]
        t = np.array(t)
        t = np.apply_along_axis(discounted, 1, t)
        return indices, X, t, y



class FunctionTestCase(unittest.TestCase):
    def testCount(self):
        c = count(1)
        self.assertEqual([next(c) for _ in range(3)], [1,2,3])
        c = count(2.5, 0.5)
        self.assertEqual([next(c) for _ in range(3)], [2.5,3,3.5])
        c = count(1, 2, 3)
        # The use of the list() instead of [] suppress the StopIteration exception thrown by next()
        self.assertEqual(list(next(c) for _ in range(4)), [1,3,5])
        c = count(0, 1, 0)
        self.assertEqual(list(next(c) for _ in range(1)), [])

    def testDiscounted(self):    
        self.assertEqual(discounted([0.1, 0.2], gamma=0.5), 0.2)


class StockDataTestCase(unittest.TestCase):
    @classmethod
    def _d(cls):
        if not hasattr(cls, 'd'):
            cls.d = StockData('data2_test.csv',
            window=5,
            train_batch_num=2, train_batch_size=3,
            valid_batch_num=1, valid_batch_size=5,
            test_batch_num=2, test_batch_size=2,
            online_train_batch_num=1, p=1)
        return cls.d

    def test_fi(self):
        d = self._d()
        self.assertEqual(d._fi('Close'), [3])
        self.assertEqual(d._fi('Close', 'Open'), [3,0])
        self.assertEqual(d._fi('NONE'), [])
        self.assertEqual(d._fi('Open', 'NONE'), [0])

    def test_preprocess(self):
        d = self._d()
        self.assertTrue(np.all(d.data[d._fi('ROC5')]>=-1))
        self.assertAlmostEqual(d.data[d._fi('CCI')].mean(), 0)

    def test_normalize(self):
        d = self._d()
        X = d.data[:,:,0:3]
        X = np.ones(X.shape)
        o = d._fi('Open')
        v = d._fi('Volume')
        X[o] = X[o].cumsum(axis=2) # 1,2,3
        self.assertEqual(X[o][0,0,2], 3)
        X[v,0,0] = 2
        X[v] = X[v].cumsum(axis=2) # 2,3,4
        self.assertEqual(X[v][0,0,2], 4)

        X = d._normalize(X)
        self.assertEqual(X[o,0,0], 0)
        self.assertEqual(X[o,0,1], 1) # (2-1)/1
        self.assertEqual(X[o,0,2], 2) # (3-1)/1
        self.assertEqual(X[v,0,2], 1) # (4-2)/2

    def test_historical_period(self):
        d = self._d()
        t = d._historical_period(1,4)
        # cash
        self.assertEqual(t.shape[1], 11)
        self.assertTrue(np.all(t[:,0,:]==0))

    def test_price_change(self):
        d = self._d()
        y = d._price_change(1,4)
        self.assertEqual(y.shape, (3,11))
        self.assertEqual(y[0,0], 0)

    def test_get_batch(self):
        d = self._d()
        i, X, y = d._get_batch(11,14)
        self.assertEqual(X.shape, (3, len(d.features), len(d.stocks)+1, d.window))
        self.assertEqual(y.shape, (3, len(d.stocks)+1))
        self.assertEqual(i, [11, 12, 13])
        self.assertTrue((X[0, 0, 0]==[0, 0, 0, 0, 0]).all()) # cash
        o = d._fi('Open')[0]
        c = d._fi('Close')[0]
        # X is of shape (batch, feature, asset, day)
        self.assertEqual(X[0, o, 1, 0], d.data[o, 0, 11-5]/d.data[c, 0, 11-5]-1)
        self.assertEqual(X[1, o, 1, 1], d.data[o, 0, 11-5+1+1]/d.data[c, 0, 11-5+1]-1)

    def test_train(self):
        d = self._d()
        g = d.train()
        i, X, t, y = next(g) # [5,5+3)
        i, X, t, y = next(g) # [8,8+3)
        self.assertEqual(i, [8,9,10])

    def test_valid(self):
        d = self._d()
        g = d.valid()
        i, X, y = next(g) # [11, 11+5)
        self.assertEqual(i, list(range(11, 11+5)))

    def test_test(self):
        d = self._d()
        g = d.test()
        i, X, y = next(g) # [16,16+2)
        i, X, y = next(g) # [18,18+2)
        self.assertEqual(i, [18, 19])

        # test_online_train
        i2, X2, t2, y2 = next(d.online_train())
        self.assertEqual(i, i2[-2:])

    def test_UBAH(self):
        d = self._d()
        m = d.UBAH(10, 15)
        self.assertEqual(m.shape, (5,))
        self.assertFalse(np.isnan(m).any())
        c = d._fi('Close')[0]
        self.assertEqual(m[1], np.mean(d.data[c,:,10+1]/d.data[c,:,10]-1))

    def test_single_stock(self):
        d = StockData('data2_test.csv',
            cash=False,
            window=5, features=['Close'], stocks=['AAPL'],
            train_batch_num=2, train_batch_size=3,
            valid_batch_num=1, valid_batch_size=5,
            test_batch_num=2, test_batch_size=2)
        g = d.train()
        i, X, t, y = next(g) # [5,5+3)
        i, X, t, y = next(g) # [8,8+3)
        self.assertEqual(X.shape, (3, 1, 1, 5))
        # close price on the the second day of the second sequence
        self.assertEqual(X[1, 0, 0, 1],
            d.data_raw['Close', 'AAPL'][8+1-5+1]/d.data_raw['Close', 'AAPL'][8+1-5]-1)
        m = d.UBAH(10, 15)
        self.assertEqual(m[1],
            d.data_raw.iloc[10+1]['Close', 'AAPL']/d.data_raw.iloc[10]['Close', 'AAPL']-1)

    def test_return(self):
        d1 = StockData_CR('data2_test.csv',
            window=5,
            train_batch_num=2, train_batch_size=3,
            valid_batch_num=1, valid_batch_size=5,
            test_batch_num=2, test_batch_size=2,
            online_train_batch_num=1, p=1)

        d1._get_train_batch(5, 8)

        d2 = StockData_DR('data2_test.csv',
            window=5,
            train_batch_num=2, train_batch_size=3,
            valid_batch_num=1, valid_batch_size=5,
            test_batch_num=2, test_batch_size=2,
            online_train_batch_num=1, p=1)
        d2._get_train_batch(6, 9)


if __name__ == '__main__':
    unittest.main()

