# pip install -r PPO/requirements.txt

import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
from stable_baselines3.common.atari_wrappers import WarpFrame

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best_model.zip")

# Load the trained model
model = PPO.load(model_path)
# Create the environment
env = make_vec_env("CarRacing-v3", n_envs=1, wrapper_class=WarpFrame)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)
# Reset the environment
obs, info = env.reset()
# Enjoy the trained agent
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    env.render()
env.close()

