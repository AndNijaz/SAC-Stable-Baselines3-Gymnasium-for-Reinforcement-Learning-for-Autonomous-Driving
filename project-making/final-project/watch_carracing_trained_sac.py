import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

MODEL_PATH = r"final-project\best_model_sac.zip"
ENV_ID = "CarRacing-v3"
SEED = 42

def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env.reset(seed=seed)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)  # (84,84,1)
        return env
    return _init

env = DummyVecEnv([make_env(SEED)])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

print("Env obs space:", env.observation_space)

# Key line: override buffer_size so it doesn't try to allocate 1e6 transitions
model = SAC.load(
    MODEL_PATH,
    env=env,  # give it spaces
    device="cpu",
    custom_objects={
        "buffer_size": 1,                # or 10, or 1000, anything tiny
        "learning_starts": 0,
    },
)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    if dones[0]:
        obs = env.reset()
