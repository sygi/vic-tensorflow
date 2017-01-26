import gym
from gym import error, spaces, utils
from gym.utils import seeding


class TestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

