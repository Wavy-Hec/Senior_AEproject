
import gym
from stable_baselines3 import PPO
import datetime
import numpy as np

experiments = [27]
experiments = experiments.extend(list(range(26, 0, -2)))
experiments.append(1)

for x in experiments:
    # Create the environment

    env = gym.make("custom:Ant2-v4", render_mode="rgb_array", numFeatures = x, squeezerType='AE')

    # Learning rate schedule: linearly decreasing from 0.0007 to 0.0001
    def linear_lr(progress_remaining: float):
        start_lr = 0.0007
        end_lr = 0.0002
        return end_lr + (start_lr - end_lr) * progress_remaining

    # Initialize the PPO model with the learning rate schedule
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ant_tensorboard/", learning_rate=linear_lr)

    # Train the model
    experiment_name = f"enc{env.numFeatures}-hr0.1" # hr is "healthy reward"
    model.learn(total_timesteps=2_000_000, tb_log_name=f"{experiment_name}")

    # Save the model
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model.save(f"./models/{experiment_name}_{now}")
