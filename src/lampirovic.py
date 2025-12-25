from huggingface_hub import hf_hub_download
import torch as th
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.atari_wrappers import WarpFrame

from huggingface_hub import list_repo_files

repo_id = "kuds/car-racing-sac"
files = list_repo_files(repo_id=repo_id)
print("\n".join(files))


# Download the model from the Hub
model_path = hf_hub_download(repo_id="kuds/car-racing-sac", filename="best-model.zip")

# Load the model
model = SAC.load(model_path)

# Create the environment
env = make_vec_env("CarRacing-v3", n_envs=1, wrapper_class=WarpFrame)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Enjoy the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
