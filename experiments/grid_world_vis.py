from __future__ import print_function

from experiments.grid_world_exp import GridWorldExperiment
import time

n_trajectories = 3
if __name__ == "__main__":
    runner = GridWorldExperiment()
    runner.train(n_episodes=100)  # TODO: pickle/load the results of experiment
    
    for omega in xrange(runner.n_options):
        print("\noption", omega, "\n=============")
        for traj in xrange(n_trajectories):  # TODO: refactor that
            print("trajectory", traj)
            runner.env._reset()
            runner.policy.set_omega(omega)
            action = -1
            while not runner.policy.is_terminal(action):
                runner.env._render()
                action = runner.policy.get_action(
                    runner.state_hash(runner.env.state))
                runner.env.step(action)
                time.sleep(0.01)

            runner.env._render(agent_color="BLUE")
            time.sleep(1.)
