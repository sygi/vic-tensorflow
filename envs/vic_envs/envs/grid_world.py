import gym
from gym import error, spaces, utils
from gym.utils import seeding


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=(9, 5), wind_proba=0.2):
        self.board_size = board_size
        self.wind_proba = wind_proba
        self._seed()

        self._reset()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(board_size[0]), spaces.Discrete(board_size[1])))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self.np_random.uniform(0., 1.) <= self.wind_proba:
            direction = self.np_random.randint(4)

            # move with the wind
            self._move(direction)
        else:
            self._move(action)

        return self.state, 0, False, {}

    def _move(self, direction):
        movement = ACTION_MEANING[direction]
        for dim in xrange(len(self.board_size)):
            self._move_coordinate(movement[dim], dim)

    def _move_coordinate(self, direction, coordinate_id):
        self.state[coordinate_id] += direction
        self.state[coordinate_id] = min(self.board_size[coordinate_id] - 1,
                                        self.state[coordinate_id])
        self.state[coordinate_id] = max(0, self.state[coordinate_id])

    def _reset(self):
        self.state = [self.board_size[0]/2, self.board_size[1]/2]
        return self.state

    def _render(self, mode='human', close=False):
        pass

ACTION_NAME = {
    0 : "NOOP",
    1 : "UP",
    2 : "DOWN",
    3 : "RIGHT",
    4 : "LEFT",
}

ACTION_MEANING = {
    0: [0, 0],
    1: [-1, 0],
    2: [1, 0],
    3: [0, 1],
    4: [0, -1],
}
