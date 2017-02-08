from __future__ import print_function

from experiments.grid_world_exp import GridWorldExperiment
import time

n_trajectories = 3
if __name__ == "__main__":
    runner = GridWorldExperiment()
    runner.train(n_episodes=100)  # TODO: pickle/load the results of experiment
    
    for omega in xrange(runner.n_options):
        print("\noption", omega, "\n=============")
        for traj in xrange(n_trajectories):
            print("trajectory", traj)
            runner.rollout(omega, render=True)
