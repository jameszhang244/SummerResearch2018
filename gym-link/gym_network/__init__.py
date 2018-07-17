from gym.envs.registration import register

register(
    id='Network-v0',
    entry_point='gym_network.envs:NetworkEnv',
    nondeterministic=True
)
