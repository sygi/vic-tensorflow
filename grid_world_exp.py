import gym
import vic_envs
from policy import QLearningPolicy
from prior import FixedUniformDiscretePrior

n_options = 30
env = gym.make("grid-world-v0")
n_actions = env.action_space.n
n_states = reduce(lambda x,y: x*y,
                  map(lambda x: x.n, env.observation_space.spaces))

prior = FixedUniformDiscretePrior(n_options)

state_first_dim = env.observation_space.spaces[0].n
def state_hash(x):
    return x[1] * state_first_dim + x[0]

policy = QLearningPolicy(n_states, n_actions, state_hash=state_hash)

if __name__ == "__main__":
    print n_actions, n_states
