"""Watch a random agent fly LunarLander-v3 and crash.

This is the "before" picture: no learning, just random engine firings.
It also prints the 8 observation numbers so you can match them to the
table in the README.
"""

import argparse

import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(description="Random agent on LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=25, help="print observation every N frames")
    args = parser.parse_args()

    # render_mode="human" opens a game window. The agent never sees the picture,
    # only the 8 numbers in `obs`.
    env = gym.make("LunarLander-v3", render_mode="human")

    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False

        while not done:
            # Actions: 0 = do nothing, 1 = left engine, 2 = main engine, 3 = right engine
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            # terminated: landed or crashed. truncated: hit the 1000-frame time limit.
            done = terminated or truncated

            if step_count % args.print_every == 0:
                print(f"step {step_count}")
                print(f"  x={obs[0]:.2f} y={obs[1]:.2f} vx={obs[2]:.2f} vy={obs[3]:.2f}")
                print(f"  angle={obs[4]:.2f} ang_vel={obs[5]:.2f} left_leg={obs[6]:.0f} right_leg={obs[7]:.0f}")
                print(f"  reward={reward:.2f}")

        print(f"episode {episode + 1} done. steps={step_count} total_reward={total_reward:.1f}\n")

    env.close()


if __name__ == "__main__":
    main()
