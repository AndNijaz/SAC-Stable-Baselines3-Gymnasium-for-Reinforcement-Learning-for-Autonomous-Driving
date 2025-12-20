import os
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


def make_env(seed: int, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init


def main():
    seed = 42
    set_random_seed(seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_env = DummyVecEnv([make_env(seed)])
    train_env = VecTransposeImage(train_env)
    train_env = VecFrameStack(train_env, n_stack=4)

    eval_env = DummyVecEnv([make_env(seed + 1)])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="results",
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="models/checkpoints",
        name_prefix="sac_carracing",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    latest_path = "models/latest.zip"

    if os.path.exists(latest_path):
        print(f"Resuming from {latest_path}")
        model = SAC.load(latest_path, env=train_env, device="cpu")
    else:
        print("Starting from scratch")
        model = SAC(
            policy="CnnPolicy",
            env=train_env,
            buffer_size=20_000,
            batch_size=64,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            device="cpu",
            verbose=1,
            tensorboard_log="logs",
            seed=seed,
        )

    # total_timesteps = 500_000
    total_timesteps = 10_000

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        log_interval=10,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save("models/latest")
    print("Saved models/latest.zip")


if __name__ == "__main__":
    main()
