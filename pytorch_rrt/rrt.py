import torch
from time import perf_counter
from typing import Callable, List

import treelib

from pytorch_rrt.action_space import ActionSpace, Action
from pytorch_rrt.state_space import StateSpace, State
from pytorch_rrt.planner import PlannerResult, Trajectory, Planner, Visualizer


def nearest_neighbor(state_space: StateSpace, states: List[State], x_rand: State):
    # TODO use k-d tree to get log instead of linear lookup
    if not torch.is_tensor(states):
        states = torch.stack(states)
    d = state_space.distance(states, x_rand)
    x_near_idx = d.argmin()
    return states[x_near_idx], x_near_idx.item()


class KinodynamicRRT(Planner):
    def __init__(self,
                 state_space: StateSpace,
                 action_space: ActionSpace,
                 propagate: Callable[..., State],
                 trajectory_cost_function: Callable[[Trajectory, State], float],
                 satisfies_constraints: Callable[[Trajectory], bool] = None,
                 batch_size: int = 512,
                 ):
        super().__init__(state_space, action_space)
        self.trajectory_cost_function = trajectory_cost_function
        self.propagate = propagate
        self.satisfies_constraints = satisfies_constraints
        self.batch_size = batch_size

    def plan(self, start: State, goal_check: Callable[[Trajectory], bool], goal: State = None, goal_sample_prob=0.2,
             environment=None, visualizer: Visualizer = None, timeout: float = 1.0) -> PlannerResult:
        states = [start]
        actions = [None]
        # TODO consider reusing tree from before (after validating edges)
        tree = treelib.Tree()
        tree.create_node(identifier=0)

        min_cost = None
        min_cost_traj = None

        dt = 0
        t0 = perf_counter()
        while dt < timeout:
            # sample a state to extend towards, sometimes the goal if given
            if goal is not None and torch.rand(1) < goal_sample_prob:
                x_target = goal
            else:
                x_target = self.state_space.sample((1,)).view(-1)

            # nearest state in the tree
            x_near, x_near_idx = nearest_neighbor(self.state_space, states, x_target)

            random_actions = self.action_space.sample((self.batch_size,))
            x_samples = self.propagate(x_near.repeat(self.batch_size, 1), random_actions, environment=environment)
            # select action that led to min distance to target
            x_next, sample_near_idx = nearest_neighbor(self.state_space, x_samples, x_target)
            best_action = random_actions[sample_near_idx]

            trajectory = self.walk_up_tree(states, actions, tree, x_near_idx)
            trajectory.states.append(x_next)
            trajectory.actions.append(best_action)
            if self.satisfies_constraints is None or self.satisfies_constraints(trajectory):
                if visualizer is not None:
                    visualizer.draw_connect(x_near, x_next)

                # add to the tree
                new_idx = len(states)
                tree.create_node(identifier=new_idx, parent=x_near_idx)
                states.append(x_next)
                actions.append(best_action)

                cost = self.trajectory_cost_function(trajectory, goal)
                if min_cost is None or cost < min_cost:
                    min_cost = cost
                    min_cost_traj = trajectory
                if goal_check(trajectory):
                    dt = perf_counter() - t0
                    return PlannerResult(trajectory=trajectory, cost=cost, dt=dt, num_nodes=len(states),
                                         reached_goal=True)
            dt = perf_counter() - t0

        return PlannerResult(trajectory=min_cost_traj, cost=min_cost, dt=dt, num_nodes=len(states), reached_goal=False)

    def walk_up_tree(self, all_states: List[State], all_actions: List[Action], tree, idx):
        traj_states = []
        traj_actions = []
        while True:
            x = all_states[idx]
            u = all_actions[idx]
            traj_states.append(x)
            if u is not None:
                traj_actions.append(u)

            # go up the tree
            parent = tree.parent(idx)
            if parent is None:
                break
            else:
                idx = parent.identifier

        traj_states.reverse()
        traj_actions.reverse()
        trajectory = Trajectory(states=traj_states, actions=traj_actions)
        return trajectory
