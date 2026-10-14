"""Train PPO on LunarLander-v3, optionally with a custom reward mode.

Examples (run from the repo root):
    python src/train.py --mode none  --name baseline        # built-in reward
    python src/train.py --mode time  --seed 0
    python src/train.py --mode hover --seed 0               # the reward-hacking run
    python src/train.py --mode fixed --seed 0               # the fix

The trained model is saved to models/<name>.zip.
"""

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from reward_wrapper import VALID_MODES, CustomReward

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--mode", default="none", choices=VALID_MODES, help="reward mode")
    parser.add_argument("--steps", type=int, default=1_000_000, help="total frames to learn from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="model file name (default: ppo_<mode>_seed<seed>)")
    args = parser.parse_args()

    name = args.name or f"ppo_{args.mode}_seed{args.seed}"
    out_path = ROOT / "models" / name
    out_path.parent.mkdir(exist_ok=True)

    # 16 copies of the game run side by side, so PPO collects 16 frames per tick.
    # Our wrapper is applied to each copy. Note: Stable-Baselines3 places its
    # Monitor UNDER our wrapper, so the ep_rew_mean in the training log is the
    # built-in reward only, not our shaped reward.
    env = make_vec_env(
        "LunarLander-v3",
        n_envs=16,
        seed=args.seed,
        wrapper_class=CustomReward,
        wrapper_kwargs={"mode": args.mode},
    )

    model = PPO(
        policy="MlpPolicy",   # small neural net: 8 numbers in, two layers of 64, 4 action scores out
        env=env,
        n_steps=1024,         # frames per copy before each update: 16 * 1024 = 16,384 per iteration
        batch_size=64,        # size of each mini-batch used for gradient steps
        n_epochs=4,           # passes over each batch of collected frames
        gamma=0.999,          # discount: how much future reward matters (high, since landing pays at the end)
        gae_lambda=0.98,      # bias/variance trade-off of the advantage estimate
        ent_coef=0.01,        # small bonus for randomness, keeps the policy exploring
        seed=args.seed,
        device="cpu",         # tiny network: CPU is as fast as the GPU/MPS here
        verbose=1,
    )

    start = time.time()
    model.learn(total_timesteps=args.steps)
    print(f"training time: {(time.time() - start) / 60:.1f} min")

    model.save(out_path)
    print(f"saved {out_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
