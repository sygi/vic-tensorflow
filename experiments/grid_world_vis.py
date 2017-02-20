from __future__ import print_function

from experiments.grid_world_exp import GridWorldExperiment
import time

n_trajectories = 3
if __name__ == "__main__":
    runner = GridWorldExperiment()
    runner.train(n_episodes=1000)  # TODO: pickle/load the results of experiment
    
    x = raw_input("Enter to proceed")
    for omega in xrange(runner.n_options):
        print("\noption", omega, "\n=============")
        for traj in xrange(n_trajectories):
            print("epsilon", runner.policy.epsilon)
            runner.rollout(omega, render=True)
            print("trajectory", traj, "final state", runner.env.state)
