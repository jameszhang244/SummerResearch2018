from gym.envs.registration import register

register(
    id='Link-v0',
    entry_point='gym_link.envs:LinkEnv',
    nondeterministic=True
)
