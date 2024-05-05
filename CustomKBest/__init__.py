# stuff
from gym.envs.registration import register
register(
    id='Ant2-v7',
    entry_point= 'custom.envs:Ant2Env'
    # max_episodes_steps = 300
    )
