"""
=============================================================================
Agro-Spectra | MODULE 3: RL TRAINING LOOP
File        : train_agent.py
Description : Trains a PPO (Proximal Policy Optimisation) agent on AgroEnv
              using Stable-Baselines3. Saves the best model during training
              via a custom callback and the final model as agro_ppo_model.zip.
              After training, evaluates the agent for one 30-day episode.
=============================================================================
Prerequisites (install via pip):
    pip install gymnasium stable-baselines3 numpy pandas

Run:
    python data_generator.py   # must run first to create mock_farm_data.csv
    python train_agent.py
=============================================================================
"""

import os
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# Import our custom environment
from agro_env import AgroEnv, MAX_STEPS

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
LOG_DIR         = os.path.join(BASE_DIR, "logs")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_agro_model")
FINAL_MODEL_ZIP = os.path.join(BASE_DIR, "agro_ppo_model")   # .zip added by SB3
TOTAL_TIMESTEPS = 50_000
EVAL_FREQ       = 2_000    # evaluate every N training steps
N_EVAL_EPISODES = 5        # episodes to average over during evaluation
SEED            = 42

os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------
# Register AgroEnv under a versioned ID so it can be created via gym.make()
# and is discoverable by OpenEnv.
AGRO_ENV_ID = "AgroEnv-v1"

# Guard against re-registration if this module is imported multiple times
_registered_ids = {spec.id for spec in gym.envs.registry.values()}
if AGRO_ENV_ID not in _registered_ids:
    register(
        id=AGRO_ENV_ID,
        entry_point="agro_env:AgroEnv",
        max_episode_steps=MAX_STEPS,
        kwargs={"seed": SEED},
    )


# ---------------------------------------------------------------------------
# Custom Callback: Save the model whenever a new best mean reward is reached
# ---------------------------------------------------------------------------
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Monitors mean training reward (from VecMonitor episode_rewards) and
    saves the model whenever a new best is achieved.

    This operates on the *training* environment reward, not a separate eval
    env — making it lightweight and suitable for resource-constrained runs.

    Parameters
    ----------
    check_freq : int
        How often (in training steps) to check for a new best.
    save_path : str
        Directory where the best model checkpoint is saved.
    verbose : int
        Verbosity level (0 = silent, 1 = print on new best).
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq   = check_freq
        self.save_path    = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """Ensure the save directory exists."""
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called after every environment step during training.

        Returns
        -------
        bool
            Always True — training continues regardless.
        """
        if self.n_calls % self.check_freq == 0:
            # VecMonitor stores episode info in self.model.ep_info_buffer
            if len(self.model.ep_info_buffer) > 0:
                # ep_info_buffer entries contain {"r": reward, "l": length, "t": time}
                episode_rewards = [ep_info["r"]
                                   for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(episode_rewards)

                if self.verbose >= 1:
                    print(
                        f"  [Callback] Step {self.num_timesteps:>6d} | "
                        f"Mean reward: {mean_reward:7.2f} | "
                        f"Best so far: {self.best_mean_reward:7.2f}"
                    )

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    checkpoint_path = os.path.join(
                        self.save_path, "best_model"
                    )
                    self.model.save(checkpoint_path)
                    if self.verbose >= 1:
                        print(
                            f"  [Callback] ** New best model saved -> "
                            f"{checkpoint_path}.zip"
                        )
        return True   # continue training


# ---------------------------------------------------------------------------
# Environment factory helpers
# ---------------------------------------------------------------------------

def make_training_env(n_envs: int = 4) -> VecMonitor:
    """
    Create a vectorised, monitored training environment.

    Parameters
    ----------
    n_envs : int
        Number of parallel environment instances.

    Returns
    -------
    VecMonitor
        Wrapped vectorised environment ready for SB3 training.
    """
    def _make_single_env():
        env = gym.make(AGRO_ENV_ID)
        env = Monitor(env)      # wraps each env for episode stats
        return env

    vec_env = DummyVecEnv([_make_single_env for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)   # aggregates Monitor stats across envs
    return vec_env


def make_eval_env() -> Monitor:
    """
    Create a single monitored evaluation environment.

    Returns
    -------
    Monitor-wrapped AgroEnv
    """
    env = gym.make(AGRO_ENV_ID)
    return Monitor(env)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> PPO:
    """
    Initialise and train a PPO agent on AgroEnv.

    Training setup:
    - Policy    : MlpPolicy (two hidden layers, default 64 units each)
    - Algorithm : PPO with default SB3 hyper-parameters
    - Callbacks : SaveOnBestTrainingRewardCallback + EvalCallback
    - Timesteps : 50,000

    Returns
    -------
    PPO
        The trained model.
    """
    print("\n" + "=" * 60)
    print("  Agro-Spectra | PPO Training")
    print("=" * 60)

    # ---- Environments ------------------------------------------------------
    train_env = make_training_env(n_envs=4)
    eval_env  = make_eval_env()

    # ---- Model -------------------------------------------------------------
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        verbose         = 1,
        seed            = SEED,
        tensorboard_log = None,    # set to LOG_DIR if tensorboard is installed
        # Hyper-parameters tuned for short 30-step episodes
        n_steps         = 128,     # rollout buffer size per env
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.97,    # slightly lower discount — short horizon
        learning_rate   = 3e-4,
        clip_range      = 0.2,
        ent_coef        = 0.01,    # encourage exploration early on
    )

    print(f"\nModel architecture:\n{model.policy}\n")

    # ---- Callbacks ---------------------------------------------------------
    best_model_callback = SaveOnBestTrainingRewardCallback(
        check_freq = 500,
        save_path  = BEST_MODEL_PATH,
        verbose    = 1,
    )

    eval_callback = EvalCallback(
        eval_env          = eval_env,
        n_eval_episodes   = N_EVAL_EPISODES,
        eval_freq         = EVAL_FREQ,
        log_path          = LOG_DIR,
        best_model_save_path = BEST_MODEL_PATH,
        deterministic     = True,
        verbose           = 1,
    )

    callback_list = CallbackList([best_model_callback, eval_callback])

    # ---- Train -------------------------------------------------------------
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps ...\n")
    model.learn(
        total_timesteps    = TOTAL_TIMESTEPS,
        callback           = callback_list,
        reset_num_timesteps= True,
    )

    # ---- Save final model --------------------------------------------------
    model.save(FINAL_MODEL_ZIP)
    print(f"\n[OK] Final model saved -> {FINAL_MODEL_ZIP}.zip")

    train_env.close()
    eval_env.close()

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: PPO) -> None:
    """
    Run one deterministic 30-day episode and log per-step actions and rewards.

    Parameters
    ----------
    model : PPO
        Trained (or loaded) PPO model.
    """
    action_labels = {0: "Do Nothing", 1: "Irrigate  ", 2: "Fertilize "}

    print("\n" + "=" * 60)
    print("  Agro-Spectra | 30-Day Evaluation Episode")
    print("=" * 60)
    print(f"\n{'Day':>3} | {'Action':>10} | {'Reward':>7} | "
          f"{'Moisture':>8} | {'Nitrogen':>8} | {'Temp C':>6} | {'Rain mm':>7}")
    print("-" * 65)

    # Use a fresh, unseeded env for the evaluation episode
    eval_env = AgroEnv()
    obs, info = eval_env.reset()

    episode_reward   = 0.0
    optimal_days     = 0
    step             = 0

    while True:
        # Deterministic prediction from the trained policy
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        step += 1

        if info["in_optimal_band"]:
            optimal_days += 1

        print(
            f"{step:>3d} | {action_labels[action]:>10} | {reward:>7.2f} | "
            f"{info['moisture']:>8.1f} | {info['nitrogen']:>8.1f} | "
            f"{info['temperature']:>6.1f} | {info['rainfall']:>7.1f}"
        )

        if terminated or truncated:
            break

    eval_env.close()

    print("-" * 65)
    print(f"\n  **  Episode Summary")
    print(f"     Total Reward   : {episode_reward:>8.2f}")
    print(f"     Days in optima : {optimal_days:>3d} / {step}")
    print(f"     Mean Reward/Day: {episode_reward / step:>8.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # STEP 1 — Sanity check: make sure data exists
    csv_path = os.path.join(BASE_DIR, "mock_farm_data.csv")
    if not os.path.exists(csv_path):
        print("[train_agent] mock_farm_data.csv not found.")
        print("[train_agent] Running data_generator.py automatically ...\n")
        import subprocess, sys
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "data_generator.py")],
            check=True
        )

    # STEP 2 — Train
    trained_model = train()

    # STEP 3 — Evaluate
    evaluate(trained_model)

    print("\n[All done] Agro-Spectra training complete.")
    print(f"  Final model   : {FINAL_MODEL_ZIP}.zip")
    print(f"  Best model    : {BEST_MODEL_PATH}/best_model.zip")
    print(f"  TensorBoard   : tensorboard --logdir {LOG_DIR}")
