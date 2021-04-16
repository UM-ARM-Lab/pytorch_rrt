from abc import ABC

import torch
from typing import Iterable
from collections import namedtuple
from torch.distributions.uniform import Uniform

Action = torch.tensor
ActionDescription = namedtuple('ActionDescription', ['name', 'min', 'max'])


class ActionSpace:
    @classmethod
    def description(cls) -> Iterable[ActionDescription]:
        """Description of each action dimension"""
        raise NotImplementedError()

    @classmethod
    def dim(cls):
        return len(list(cls.description()))

    def similarity(self, u1: Action, u2: Action) -> torch.tensor:
        """Measure of similarity between two actions, [0, 1]"""
        return torch.zeros(u1.shape[0], dtype=u1.dtype, device=u1.device)

    def sample(self, batch_size) -> Action:
        raise NotImplementedError()


class UniformActionSpace(ActionSpace, ABC):
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.d = device
        self.u_min = torch.tensor([desc.min for desc in self.description()], dtype=self.dtype, device=self.d)
        self.u_max = torch.tensor([desc.max for desc in self.description()], dtype=self.dtype, device=self.d)
        self.dist = Uniform(self.u_min, self.u_max)
        self._dim = super().dim()

    def dim(self):
        return self._dim  # cached dim

    def sample(self, batch_size) -> Action:
        return self.dist.sample(batch_size)
