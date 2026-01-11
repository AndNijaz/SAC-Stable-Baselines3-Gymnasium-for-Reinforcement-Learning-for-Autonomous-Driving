import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

MODEL_PATH = r"final-project\best_model_ppo.zip"
ENV_ID = "CarRacing-v3"
SEED = 42

def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env.reset(seed=seed)

        # Must match training preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

        return env
    return _init

env = DummyVecEnv([make_env(SEED)])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

print("Env obs space:", env.observation_space)

model = PPO.load(MODEL_PATH, device="cpu")
model.set_env(env)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    if dones[0]:
        obs = env.reset()
