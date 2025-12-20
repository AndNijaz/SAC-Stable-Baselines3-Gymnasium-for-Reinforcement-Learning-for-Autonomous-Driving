import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

MODEL_PATH = "models/latest"   # SB3 adds .zip automatically
ENV_ID = "CarRacing-v3"
SEED = 42


def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env.reset(seed=seed)
        return env
    return _init


# --- Create env EXACTLY like training ---
env = DummyVecEnv([make_env(SEED)])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

# --- Load model ---
model = SAC.load(MODEL_PATH, env=env)

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

    if dones[0]:
        obs = env.reset()
