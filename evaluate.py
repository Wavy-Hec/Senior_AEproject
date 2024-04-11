
import gym
from stable_baselines3 import PPO
from time import sleep

# Create the environment

env = gym.make("custom:Ant2-v4", render_mode="rgb_array")

# Load the model
try:
    model = PPO.load("models/enc27_20240322_1345.zip", env=env)
except Exception as e:
    print(e)
    exit()

# Interact with the environment post-training for visualization
vec_env = model.get_env()
obs = vec_env.reset()

sumRew = 0
for i in range(1000):  # loop simulating 10000 timesteps
    sleep(1/60) # take an action 60fps
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)
    sumRew += rewards
    print(i, sumRew)
    if done:
            obs = vec_env.reset()
    
    vec_env.render("human")