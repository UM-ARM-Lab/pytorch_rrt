from abc import ABC

import torch
from typing import Iterable
from collections import namedtuple
from torch.distributions.uniform import Uniform

State = torch.tensor
StateDescription = namedtuple('StateDescription', ['name', 'min', 'max'])


class StateSpace:
    @classmethod
    def description(cls) -> Iterable[StateDescription]:
        """Description of each state dimension"""
        raise NotImplementedError()

    @classmethod
    def dim(cls):
        return len(list(cls.description()))

    def distance(self, s1: State, s2: State) -> torch.tensor:
        raise NotImplementedError()

    def sample(self, batch_size) -> State:
        raise NotImplementedError()


class UniformStateSpace(StateSpace, ABC):
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.d = device
        self.u_min = torch.tensor([desc.min for desc in self.description()], dtype=self.dtype, device=self.d)
        self.u_max = torch.tensor([desc.max for desc in self.description()], dtype=self.dtype, device=self.d)
        self.dist = Uniform(self.u_min, self.u_max)
        self._dim = super().dim()

    def dim(self):
        return self._dim  # cached dim

    def sample(self, batch_size) -> State:
        return self.dist.sample(batch_size)
