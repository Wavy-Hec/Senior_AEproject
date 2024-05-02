
import gym
from stable_baselines3 import SAC
import datetime
import numpy as np
from custom.envs.humanoid_v4 import *

# Create the environment

# env = gym.make([Human2Env], render_mode="rgb_array")
env = Human2Env(numFeatures=200)

# Learning rate schedule: linearly decreasing from 0.0007 to 0.0001
def linear_lr(progress_remaining: float):
    start_lr = 0.0007
    end_lr = 0.0002
    return end_lr + (start_lr - end_lr) * progress_remaining

# Initialize the PPO model with the learning rate schedule
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_human_tensorboard/", learning_rate=linear_lr)

# Load the model
try:
    model.load("./models/human")
except Exception as e:
    print(e)
    print("Starting a new model")

# Train the model
experiment_name = f"enc{env.numFeatures}"
model.learn(total_timesteps=1_000, tb_log_name=f"{experiment_name}")

# Save the model
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model.save(f"./models/{experiment_name}_{now}")

# Interact with the environment post-training for visualization
vec_env = model.get_env()
obs = vec_env.reset()

for i in range(10_000):  # loop simulating 10000 timesteps
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)

    vec_env.render("human")
    if done:
        obs = vec_env.reset()