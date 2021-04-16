from pytorch_rrt import UniformActionSpace, ActionDescription, \
    UniformStateSpace, State, StateDescription, \
    KinodynamicRRT, Visualizer
from typing import Iterable
import torch
# from window_recorder.recorder import WindowRecorder

import matplotlib.pyplot as plt


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


class TwoDVisualizer(Visualizer):
    def draw_state(self, x: State, color='k', s=4, alpha=0.2):
        x = x.cpu().numpy()
        plt.scatter(x[0], x[1], color=color, s=s, alpha=alpha)
        # plt.pause(0.0001)

    def draw_connect(self, x_start: State, x_next: State):
        self.draw_state(x_next)
        plt.plot([x_start[0].cpu().numpy(), x_next[0].cpu().numpy()],
                 [x_start[1].cpu().numpy(), x_next[1].cpu().numpy()], color='gray', linewidth=1, alpha=0.2)
        plt.pause(0.0001)


state_space = TwoDStateSpace()
action_space = TwoDActionSpace()


def true_dynamics(state, action, environment=None):
    return state + action


# try different true dynamics than given approximate dynamics
dynamics = true_dynamics


def traj_cost(trajectory, goal):
    states = torch.stack(trajectory.states)
    d = state_space.distance(states, goal)
    return d.min()


rrt = KinodynamicRRT(state_space, action_space, dynamics, traj_cost)

goal = state_space.sample((1,)).view(-1)
state = state_space.sample((1,)).view(-1)


def goal_check(trajectory):
    states = torch.stack(trajectory.states)
    d = state_space.distance(states, goal)
    return d.min() < 0.1


vis = TwoDVisualizer()
plt.ion()

plt.figure()
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.axis("equal")
vis.draw_state(state, color='k', s=20, alpha=1)
vis.draw_state(goal, color='g', s=20, alpha=1)
plt.draw()

# use RRT in MPC manner, re-plan after each action
while True:
    res = rrt.plan(state, goal_check, goal=goal, visualizer=vis)
    action = res.trajectory.actions[0]
    # step in environment

    next_state = true_dynamics(state, action)
    state = next_state

    vis.draw_state(state, color='k', s=8, alpha=1)
    plt.draw()

    if state_space.distance(state, goal) < 0.1:
        print("done planning state: {} goal: {}".format(state, goal))
        break

input('enter to finish')
