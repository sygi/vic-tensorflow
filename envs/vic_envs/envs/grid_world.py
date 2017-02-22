import gym
import pyglet

from gym import error, spaces, utils
from gym.utils import seeding


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=(5, 9), wind_proba=0.2, stay_wind=True):
        self.board_size = board_size
        self.wind_proba = wind_proba
        self.stay_wind = stay_wind
        self._seed()

        self._reset()
        self.action_space = spaces.Discrete(len(ACTION_MEANING))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(board_size[0]), spaces.Discrete(board_size[1])))
        self.window = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if ACTION_MEANING[action] == 'NOOP' and self.stay_wind == False:
            self._move(action)
        else:
            if self.np_random.uniform(0., 1.) <= self.wind_proba:
                direction = self.np_random.randint(1, 6)

                # move with the wind
                self._move(direction)
            else:
                self._move(action)

        return self.state, 0, ACTION_MEANING[action] == "FINISH", {}

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

    def _render(self, mode='human', close=False, agent_color="ORANGE"):
        self.unit = 30
        if close:
            if self.window is not None:
                self.window.close()
            return

        if self.window is None:
            self.window = pyglet.window.Window(height=(self.board_size[1] *
                                                       self.unit),
                                               width=(self.board_size[0] *
                                                      self.unit))
        self.window.dispatch_events()

        for x in xrange(self.board_size[0]):
            for y in xrange(self.board_size[1]):
                self._draw_field(x, y)

        self._draw_agent(*self.state, color=agent_color)
        self.window.flip()
 
    def _draw_field(self, x, y):
        if (x + y) % 2 == 0:
            color = ('c3B', (224, 224, 224) * 4)  # light gray
        else:
            color = ('c3B', (160, 160, 160) * 4)  # gray

        vertex_list = ([(x_it * self.unit, y * self.unit)
                        for x_it in xrange(x, x+2)] +
                       [(x_it * self.unit, (y+1) * self.unit)
                        for x_it in xrange(x+1, x-1, -1)])

        vertex_flat = [coord for vert in vertex_list for coord in vert]

        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', vertex_flat),
                             color)

    def _draw_agent(self, x_agent, y_agent, color="ORANGE"):
        xs = [self.unit/4, (3*self.unit)/4, self.unit/2]
        ys = [self.unit/5, self.unit/5, (4*self.unit)/5]

        vertex_moved = zip([x_agent * self.unit + x for x in xs],
                           [y_agent * self.unit + y for y in ys])

        vertex_flat = [coord for vert in vertex_moved for coord in vert]

        if color == "ORANGE":
            color_vert = ('c3B', (255, 153, 51) * 3)
        elif color == "BLUE":
            color_vert = ('c3B', (51, 153, 255) * 3)
        else:
            raise NotImplementedError
        pyglet.graphics.draw(3, pyglet.gl.GL_TRIANGLES, ('v2f', vertex_flat),
                             color_vert)


ACTION_NAME = {
    0 : "FINISH",
    1 : "NOOP",
    2 : "LEFT",
    3 : "RIGHT",
    4 : "UP",
    5 : "DOWN",
}

ACTION_MEANING = {
    0: [0, 0],
    1: [0, 0],
    2: [-1, 0],
    3: [1, 0],
    4: [0, 1],
    5: [0, -1],
}
