"""Reward-shaping wrapper for LunarLander-v3.

A Gymnasium wrapper is a thin layer around an environment. It can change what
comes out of `step()` (the reward, the observation, ...) without touching the
environment's own code. PPO talks to the wrapper, the wrapper talks to the game.

    PPO  <-->  CustomReward  <-->  LunarLander-v3

Four modes let us run the whole experiment with one class:

    "none"   built-in reward only (the baseline)
    "time"   built-in reward minus a small fee every frame
    "hover"  built-in reward plus +1 per frame while upright and slow
             (a deliberately BAD reward, used to show reward hacking)
    "fixed"  time fee plus a potential-based "upright and slow" signal
             (the fix: pay for improving, not for sitting in a nice spot)
"""

import gymnasium as gym

VALID_MODES = ("none", "time", "hover", "fixed")

# Thresholds for "upright and slow". Values are in the environment's own
# normalised observation units, not radians or metres per second.
MAX_TILT = 0.2   # observation[4]: tilt angle
MAX_SPEED = 0.3  # observation[2], observation[3]: sideways / vertical speed


def is_upright_slow(obs):
    """Return True when the lander is nearly upright and moving slowly.

    Observation layout (8 numbers):
        0 x position        4 tilt angle
        1 y position        5 angular velocity
        2 x velocity        6 left leg touching ground (0/1)
        3 y velocity        7 right leg touching ground (0/1)
    """
    return (
        abs(obs[4]) < MAX_TILT
        and abs(obs[2]) < MAX_SPEED
        and abs(obs[3]) < MAX_SPEED
    )


class CustomReward(gym.Wrapper):
    """Adds our own reward terms on top of the built-in LunarLander reward."""

    def __init__(self, env, mode="none", time_fee=0.05, gamma=0.999):
        super().__init__(env)
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.time_fee = time_fee  # points removed every frame ("waiting costs something")
        self.gamma = gamma        # must match PPO's gamma for the "fixed" mode maths
        self.prev_phi = 0.0       # potential of the previous frame ("fixed" mode only)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Start-of-episode potential, so the first frame is not rewarded for
        # "becoming" upright and slow when it already was.
        self.prev_phi = 1.0 if is_upright_slow(obs) else 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["builtin_reward"] = reward  # keep the original number for logging

        if self.mode == "time":
            # Constant pressure to finish quickly. Built-in reward has no such term.
            reward -= self.time_fee

        elif self.mode == "hover":
            # BAD REWARD ON PURPOSE. +1 for every frame spent upright and slow.
            # An episode is cut off at 1000 frames, so hanging in the air can earn
            # up to +1000, while fuel costs at most 1000 * 0.3 = 300.
            # Landing ends the episode and the +1 payments stop, so hovering wins.
            if is_upright_slow(obs):
                reward += 1.0

        elif self.mode == "fixed":
            # 1) small fee per frame, so waiting costs something
            reward -= self.time_fee
            # 2) potential-based shaping: reward the CHANGE in a potential phi,
            #    not the state itself. F = gamma * phi(next) - phi(current).
            #    Staying in the good state pays gamma*1 - 1 = -0.001 per frame,
            #    which is nothing, so there is nothing to farm by hovering.
            #    Terminal states get potential 0 so the sum telescopes cleanly.
            new_phi = 0.0 if terminated else (1.0 if is_upright_slow(obs) else 0.0)
            reward += self.gamma * new_phi - self.prev_phi
            self.prev_phi = new_phi

        return obs, reward, terminated, truncated, info
