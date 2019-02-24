import gym
from gym import spaces
import numpy as np
import pandas as pd
from dataset import Dataset

class PortfolioEnv(gym.Env):
    def __init__(self, features, stocks, batch_num=1, batch_size=10, window=10):
        self.features = features
        self.stocks = stocks
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.window = window # using historical price in the past M days as feature
        self.assets = len(self.stocks) + 1
        self.observation_shape = (len(self.features), self.assets, self.window)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape),
            high=np.full(self.observation_shape, np.finfo(np.float32).max),
            dtype=np.float32)

        self.data = Dataset('data.csv', features=self.features, stocks=self.stocks,
                        train_batch_num=self.batch_num,
                        batch_size=self.batch_size,
                        window=self.window)
        self._step = self._step_generator()

    def _step_generator(self):
        while True: # loop
            for X, y in self.data.train_batch(): # episode
                for i in range(self.batch_size+1): # step
                    if self._reset:
                        self._reset = False
                        if i: # not at the begining, move to next batch
                            break
                    if i == self.batch_size:
                        self.done = True
                        self.state = None
                        self.y = None
                    else:
                        self.done = False
                        self.state = X[i]
                        self.y = y[i]
                    yield

    def step(self, w):
        self._record = np.array([self.y-1, w]) # # record for rendering
        if self.done:
            raise RuntimeError('Current episode has already been done. Call reset to start another episode.')
        w = np.clip(w, 0, 1)
        reward = np.log(np.sum(w * self.y))
        next(self._step)
        return self.state, reward, self.done, {}

    def reset(self):
        '''reset can be called at any step:
        1. it will be called at the begining of any episode
        2. it might also be called during an episode
        
        This method results in moving to next batch
        '''
        self._reset = True
        next(self._step)
        return self.state

    def render(self, mode='human'):
        '''first line is price relative change rate, second line is portfolio weights'''
        print(self._record)

    
from gym.envs.registration import register
gym.envs.registry.env_specs.pop('Portfolio-v0', None)
register(
    id='Portfolio-v0',
    entry_point=PortfolioEnv
)


if __name__ == '__main__':
    from numpy.testing import assert_equal
    w = [1, 0]
    env = gym.make('Portfolio-v0',
                   features=['Close'],
                   stocks = ['GOOGL'],
                   batch_num = 1,
                   batch_size = 10,
                   window=10)
    obsv = env.reset()
    for i in range(9):
        state, reward, done, _ = env.step(w)
        assert_equal(done, False)
    state, reward, done, _ = env.step(w)
    assert_equal(done, True)
    assert_equal(state, None)
    obsv2 = env.reset()
    assert_equal(obsv, obsv2)

    env = gym.make('Portfolio-v0',
                   features=['Close'],
                   stocks = ['GOOGL'],
                   batch_num = 2,
                   batch_size = 10,
                   window=10)
    obsv = env.reset()
    for i in range(10):
        env.step(w)
    env.reset()
    state, reward, done, _ = env.step(w)
    obsv2 = env.reset()
    assert_equal(obsv, obsv2)
