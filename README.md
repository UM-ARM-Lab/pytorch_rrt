# PyTorch Kinodynamic RRT Implementation
This repository implements Kinodynamic Rapidly-exploring Random Tree (RRT) 
with black-box dynamics and constraints in pytorch. RRT is a motion planning algorithm
that returns a trajectory.

# Usage
Clone repository somewhere, then `pip3 install -e .` to install in editable mode.
See `tests/twod.py` for example usage. 

```python
from pytorch_rrt import UniformActionSpace, ActionDescription, \
    UniformStateSpace, State, StateDescription, \
    KinodynamicRRT
from typing import Iterable
import torch

# define action and state space
class TwoDActionSpace(UniformActionSpace):
    MAX_ACTION = 0.3

    @classmethod
    def description(cls) -> Iterable[ActionDescription]:
        return [ActionDescription("dx", -cls.MAX_ACTION, cls.MAX_ACTION),
                ActionDescription("dy", -cls.MAX_ACTION, cls.MAX_ACTION)]


class TwoDStateSpace(UniformStateSpace):
    MAX_STATE = 3

    @classmethod
    def description(cls) -> Iterable[StateDescription]:
        return [StateDescription("x", -cls.MAX_STATE, cls.MAX_STATE),
                StateDescription("y", -cls.MAX_STATE, cls.MAX_STATE)]

    def distance(self, s1: State, s2: State) -> torch.tensor:
        return (s1 - s2).view(-1, self.dim()).norm(dim=1)

state_space = TwoDStateSpace()
action_space = TwoDActionSpace()

# given external dynamics and trajectory cost
rrt = KinodynamicRRT(state_space, action_space, dynamics, traj_cost)
```

Can use RRT to plan open-loop (assuming gym-like `env`)
```python
res = rrt.plan(state, goal_check, goal=goal, timeout=5.0)
actions = res.trajectory.actions
for action in actions:
    env.step(action.cpu().numpy())
```

or in a closed-loop manner
```python
while True:
    res = rrt.plan(state, goal_check, goal=goal, timeout=1.0)
    action = res.trajectory.actions[0]
    # step in environment
    state, reward, done, _  = env.step(action.cpu().numpy())
    if done:
        break
```

# Requirements
- pytorch (>= 1.0)

