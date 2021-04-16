from typing import List, Callable

from pytorch_rrt.action_space import ActionSpace, Action
from pytorch_rrt.state_space import StateSpace, State

from dataclasses import dataclass


@dataclass
class Trajectory:
    states: List[State]
    actions: List[Action]


@dataclass
class PlannerResult:
    trajectory: Trajectory
    cost: float
    dt: float
    num_nodes: int
    reached_goal: bool


class Visualizer:
    def draw_connect(self, x_start: State, x_next: State):
        return NotImplementedError()


class Planner:

    def __init__(self, state_space: StateSpace, action_space: ActionSpace):
        self.state_space = state_space
        self.action_space = action_space

    def plan(self, start: State, goal_check: Callable[[Trajectory], bool], goal: State = None, goal_sample_prob=0.2,
             environment=None, visualizer: Visualizer = None, timeout: float = 1.0) -> PlannerResult:
        raise NotImplementedError()
