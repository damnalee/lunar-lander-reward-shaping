"""Evaluate a trained model on the PLAIN LunarLander-v3 (reward wrapper OFF).

Testing without our wrapper matters: a bonus we added during training must not
flatter the score. A score of 200+ per episode counts as "solved".

Examples (run from the repo root):
    python src/evaluate.py --model models/ppo_baseline
    python src/evaluate.py --model models/ppo_hover_seed0 --record --prefix hover
    python src/evaluate.py --model models/ppo_baseline --render
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent


def run_episodes(env, model, episodes, seed=None, verbose=True):
    """Fly `episodes` attempts and return (rewards, lengths, number_of_timeouts)."""
    rewards, lengths, timeouts = [], [], 0
    for episode in range(episodes):
        obs, info = env.reset(seed=None if seed is None else seed + episode)
        total_reward, steps, done = 0.0, 0, False
        while not done:
            # deterministic=True: always take the best-scoring action instead of
            # sampling from the policy. Training samples (to explore), testing doesn't.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        timeouts += int(truncated)  # truncated = ran out of the 1000-frame limit (hovering)
        if verbose:
            print(f"episode {episode + 1}: reward={total_reward:.1f} steps={steps} timeout={truncated}")
    return rewards, lengths, timeouts


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument("--model", required=True, help="path to the model, with or without .zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducible episodes")
    parser.add_argument("--render", action="store_true", help="show the game window for 3 episodes")
    parser.add_argument("--record", action="store_true", help="save 2 episodes as .mp4 in videos/")
    parser.add_argument("--prefix", default=None, help="video file name prefix")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")

    # 1) Numbers: headless, many episodes.
    env = gym.make("LunarLander-v3")
    rewards, lengths, timeouts = run_episodes(env, model, args.episodes, args.seed)
    env.close()

    print(f"\nmean reward={np.mean(rewards):.1f} std={np.std(rewards):.1f}")
    print(f"mean steps={np.mean(lengths):.0f} timeouts={timeouts}/{args.episodes}")
    print("SOLVED" if np.mean(rewards) >= 200 else "not solved (mean reward below 200)")

    # 2) Pictures: watch it live and/or record clips. Always watch the clip,
    #    not just the score: a score can look fine while the rocket hovers.
    if args.render:
        env = gym.make("LunarLander-v3", render_mode="human")
        run_episodes(env, model, 3, args.seed, verbose=False)
        env.close()

    if args.record:
        prefix = args.prefix or Path(args.model).stem
        env = gym.make("LunarLander-v3", render_mode="rgb_array")  # rgb_array is needed for video
        env = RecordVideo(
            env,
            video_folder=str(ROOT / "videos"),
            name_prefix=prefix,
            episode_trigger=lambda episode_id: True,  # record every episode
        )
        run_episodes(env, model, 2, args.seed, verbose=False)
        env.close()
        print(f"clips saved to videos/{prefix}-episode-*.mp4")


if __name__ == "__main__":
    main()
