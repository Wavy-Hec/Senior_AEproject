from gym.envs.registration import register
register(
    id='Ant2-v4',
    entry_point= 'custom.envs:Ant2Env'
    # max_episodes_steps = 300
    )

# register(
#     id='Humanoid2-v4',
#     entry_point= 'custom.envs:Human2Env'
#     # max_episodes_steps = 300
#     )