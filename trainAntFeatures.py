import gym
from stable_baselines3 import PPO
import datetime

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Multiprocessing for envs. Sourced from SB3: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def make_env(env_id: str, numFeatures: int = 27, squeezerType: str = 'AE', rank: int = 0, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Learning rate schedule: linearly decreasing from 0.0007 to 0.0001
def linear_lr(progress_remaining: float):
    start_lr = 0.0007
    end_lr = 0.0002
    return end_lr + (start_lr - end_lr) * progress_remaining


# numCpus should be set by device
numCpus = 10
# experiments = [27] + list(range(26, 0, -2)) + [1]
experiments = [1, 2, 4] #first half of experiments

if __name__ == "__main__":
    for x in experiments:
        for i in range(10):
            print(f"Now starting experiment: {x}.{i}")
            # Create the environment
            env = gym.make("custom:Ant2-v4", numFeatures=x, squeezerType='AE')

            # # Initialize the PPO model with the learning rate schedule
            model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="logs/ant_tb/", learning_rate=linear_lr)

            # # Train the model
            experiment_name = f"AEenc{x}_{i}" # hr is "healthy reward"
            model.learn(total_timesteps=1_500_000, tb_log_name=experiment_name)

        # Save the model
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model.save(f"./models/{experiment_name}_{now}")
        print("Done")