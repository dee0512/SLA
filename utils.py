import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import os
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Optional
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            # torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            self.next_state[ind],
            # torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


class FiGARReplayBuffer(object):
    def __init__(self, state_dim, action_dim, rep_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.rep = np.zeros((max_size, rep_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, rep, next_state, next_action, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.rep[self.ptr] = rep
        self.next_state[self.ptr] = next_state
        self.next_action[self.ptr] = next_action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            # torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.rep[ind]).to(self.device),
            self.next_state[ind],
            # torch.FloatTensor(self.next_state[ind]).to(self.device),

            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )



class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param mean: the mean of the noise
    :param sigma: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
    ):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) *
            np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(
            self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"


class Clock:
    def __init__(self, clock_dim=2):
        self.state = np.zeros(clock_dim)
        self.value = 0
        self.clock_dim = clock_dim

    def tick(self):
        self.value += 1
        if self.value >= np.power(2, self.clock_dim):
            self.value = 0
        self.state = np.unpackbits(
            np.array([self.value], dtype='>i8').view(np.uint8))[-self.clock_dim:]
        return self.state

    def reset(self):
        self.value = 0
        self.state = np.unpackbits(
            np.array([self.value], dtype='>i8').view(np.uint8))[-self.clock_dim:]
        return self.state
