
import gym
from stable_baselines3 import PPO
import datetime
import numpy as np

# Create the environment

env = gym.make("custom:Ant2-v4", render_mode="rgb_array")

# Learning rate schedule: linearly decreasing from 0.0007 to 0.0001
def linear_lr(progress_remaining: float):
    start_lr = 0.0007
    end_lr = 0.0002
    return end_lr + (start_lr - end_lr) * progress_remaining

# Initialize the PPO model with the learning rate schedule
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ant_tb/", learning_rate=linear_lr)

# Load the model
try:
    model.load("./models/ant")
except Exception as e:
    print(e)
    print("Starting a new model")

# Train the model
experiment_name = f"AEenc{env.numFeatures}"
model.learn(total_timesteps=1_000, tb_log_name=f"{experiment_name}")

# Save the model
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model.save(f"./models/{experiment_name}_{now}")

# Interact with the environment post-training for visualization
vec_env = model.get_env()
obs = vec_env.reset()

for i in range(10_000):  # loop simulating 10000 timesteps
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)

    # vec_env.render("human") # Skip rendering 
    if done:
        obs = vec_env.reset()