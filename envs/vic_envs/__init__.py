from gym.envs.registration import register

register(
    id='vic-test-env-v0',
    entry_point='vic_envs.envs:TestEnv',
)

register(
    id='grid-world-v0',
    entry_point='vic_envs.envs:GridWorld',
)

register(
    id='grid-world-simpler-v0',
    entry_point='vic_envs.envs:GridWorld',
    kwargs={'stay_wind': False},
)


register(
    id='deterministic-grid-world-v0',
    entry_point='vic_envs.envs:GridWorld',
    kwargs={'wind_proba': 0.},
)

# TODO: more envs
