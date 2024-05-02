import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse

# Create directories to hold models and logs
model_dir = "CudaM"
log_dir = "CudaL"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env):
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/SAC_{TIMESTEPS*iters}")

def test(env, path_to_model):

   
    model = SAC.load(path_to_model, env=env)
      

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('-t', '--train', action='store_true', default=True)
    parser.add_argument('-s', '--test', metavar='path_to_model', default=True)
    args = parser.parse_args()


    if args.train:
        gymenv = gym.make("Humanoid-v4", render_mode=None)
        train(gymenv)

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make("Humanoid-v4", render_mode=None)
            test(gymenv, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')

