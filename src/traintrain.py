import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


class SafeActions(gym.ActionWrapper):
    def action(self, act):
        steer, gas, brake = act

        # Less aggressive steering reduces spinouts
        steer = float(np.clip(steer, -0.8, 0.8))

        # Cap throttle/brake so the agent learns control first
        gas = float(np.clip(gas, 0.0, 0.65))
        brake = float(np.clip(brake, 0.0, 0.55))

        return np.array([steer, gas, brake], dtype=np.float32)


def make_env(seed: int, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = SafeActions(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    seed = 42
    set_random_seed(seed)

    TARGET_TIMESTEPS = 1_000_000

    BUFFER_SIZE = 500_000
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4  # stability > speed
    TRAIN_FREQ = 1
    GRADIENT_STEPS = 1

    EVAL_FREQ = 25_000
    N_EVAL_EPISODES = 20  # consider 10-20 for more stable "best"

    SAVE_FREQ = 10_000

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_env = DummyVecEnv([make_env(seed)])
    train_env = VecTransposeImage(train_env)
    train_env = VecFrameStack(train_env, n_stack=4)

    # Multi-seed eval = best model is actually robust
    eval_env = DummyVecEnv([make_env(seed + i) for i in range(1, 6)])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="results",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path="models/checkpoints",
        name_prefix="sac_carracing",
        save_replay_buffer=True,   # ✅ 1) SAVE REPLAY BUFFER WITH CHECKPOINTS
        save_vecnormalize=False,
    )

    best_path = "models/best_model.zip"
    latest_path = "models/latest.zip"
    latest_rb_path = "models/latest_replay_buffer.pkl"

    loaded = False

    # ✅ 3) LOAD REPLAY BUFFER (IF EXISTS) WHEN RESUMING
    if os.path.exists(latest_path):
        print(f"Resuming from LATEST: {latest_path}")
        model = SAC.load(latest_path, env=train_env, device="cuda")

        if os.path.exists(latest_rb_path):
            print(f"Loading replay buffer: {latest_rb_path}")
            model.load_replay_buffer(latest_rb_path)
        else:
            print("No latest replay buffer found, continuing with empty buffer (not ideal).")

        loaded = True

    elif os.path.exists(best_path):
        print(f"Resuming from BEST: {best_path}")
        model = SAC.load(best_path, env=train_env, device="cuda")
        print("Note: best_model usually has no replay buffer saved, so buffer starts empty.")
        loaded = True

    else:
        print("No checkpoint found, starting from scratch")
        model = SAC(
            policy="CnnPolicy",
            env=train_env,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto_0.1",  # auto entropy (safer than fixed)
            device="cuda",
            verbose=1,
            tensorboard_log="logs",
            seed=seed,
        )

    already = model.num_timesteps if loaded else 0
    remaining = max(TARGET_TIMESTEPS - already, 0)

    print(f"Current timesteps: {already}")
    print(f"Target timesteps:  {TARGET_TIMESTEPS}")
    print(f"Will train for:    {remaining} more steps")

    if remaining == 0:
        print("Nothing to do, already at or above target timesteps.")
        return

    model.learn(
        total_timesteps=remaining,
        reset_num_timesteps=not loaded,
        log_interval=10,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # ✅ 2) SAVE LATEST MODEL + REPLAY BUFFER
    model.save("models/latest")
    model.save_replay_buffer(latest_rb_path)
    print("Saved models/latest.zip + models/latest_replay_buffer.pkl")


if __name__ == "__main__":
    main()
