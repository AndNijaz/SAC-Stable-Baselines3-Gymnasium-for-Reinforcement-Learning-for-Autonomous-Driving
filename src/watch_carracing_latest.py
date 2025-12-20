import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

MODEL_PATH = "models/latest"
ENV_ID = "CarRacing-v3"
SEED = 42

def make_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env.reset(seed=SEED)
        return env
    return _init

def main():
    # Vectorized env
    env = DummyVecEnv([make_env()])
    
    # MUST match training
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = SAC.load(MODEL_PATH, env=env)

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
