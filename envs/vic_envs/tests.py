import gym
import vic_envs

def all_installed_test():
    env = gym.make("vic-test-env-v0")

def deterministic_grid_test():
    env = gym.make("deterministic-grid-world-v0")
    prev_state = env.state
    for _ in xrange(100): env.step(0)  # noop
    assert env.state == prev_state

    while env.state[0] > 0:
        env.step(1)

    assert env.state[0] == 0
    env.step(1)
    assert env.state[0] == 0

    while env.state[1] < env.board_size[1] - 1:
        env.step(3)

    assert env.state[1] == env.board_size[1] - 1
    env.step(3)
    assert env.state[1] == env.board_size[1] - 1

