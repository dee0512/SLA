import gym
import numpy as np
import torch
# from gym.envs.mujoco import inverted_pendulum
import types
import os
import random
# from ray.rllib.env.atari_wrappers import wrap_deepmind, WarpFrame
__all__ = ["make_env", "create_folders", 'make_env_cc']

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecisonWrapper(gym.Wrapper):
    def __init__(self, env, decisions):
        super().__init__(env)
        self.decisions = decisions
        self.decisions_left = decisions

    def step(self, action, decision=True):
        obs, reward, done, info = self.env.step(action)
        self.decisions_left -= decision
        if self.decisions_left == 0:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.decisions_left = self.decisions
        return self.env.reset()


class ClockWrapper(gym.Wrapper):
    def __init__(self, env, clock_dim):
        super().__init__(env)
        self.clock_dim = clock_dim
        self.clock = utils.Clock(clock_dim)
        ob_space = self.observation_space
        low = ob_space.low
        low = np.insert(low, -1, axis=0, values=np.zeros(clock_dim))
        high = ob_space.high
        high = np.insert(high, -1, axis=0, values=np.ones(clock_dim))
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(ob_space.shape[0]+clock_dim,))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = np.append(obs, self.clock.tick(), axis=0)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        obs = np.append(obs, self.clock.reset(), axis=0)
        return obs


class PrevActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        ob_space = self.observation_space
        low = ob_space.low
        low = np.insert(low, -1, axis=0, values=[self.action_space.low[0]])
        high = ob_space.high
        high = np.insert(high, -1, axis=0, values=[self.action_space.high[0]])
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(ob_space.shape[0] + 1,))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = np.append(obs, action, axis=0)
        return obs, reward, done, info

    def reset(self):
        obs, done = super().reset(), False
        obs = np.append(obs, np.zeros(self.action_space.shape), axis=0)
        return obs


def make_env_cc(env_name, seed, env_timestep, decision_wrapper=False, decisions=0, clock_wrapper=False, clock_dim=0, prev_action_wrapper=False):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.dt = env_timestep
    env.env.dt = env_timestep
    if decision_wrapper:
        env = DecisonWrapper(env, decisions)

    if clock_wrapper:
        env = ClockWrapper(env, clock_dim)

    if prev_action_wrapper:
        env = PrevActionWrapper(env)

    return env


# Make environment using its name
def make_env(env_name, seed, clock_wrapper=False, clock_dim=0, prev_action_wrapper=False):
    env = gym.make(env_name)
    if clock_wrapper:
        env = ClockWrapper(env, clock_dim)

    if prev_action_wrapper:
        env = PrevActionWrapper(env)

    env._max_episode_steps = 1000
    env.seed(seed)
    env.action_space.seed(seed)

    return env


def create_folders():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")
