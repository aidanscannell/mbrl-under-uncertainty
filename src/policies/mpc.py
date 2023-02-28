#!/usr/bin/env python3
from typing import Optional

import torch
from src.custom_types import State, Action, ReplayBuffer

from .base import Policy


class MPCPolicy(Policy):
    def __init__(
        self, state_dim: int, action_dim: int, trajectory_optimizer, *args, **kwargs
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.trajectory_optimizer = trajectory_optimizer

    @torch.jit.export
    def update(self, replay_buffer: ReplayBuffer, *args, **kwargs):
        """Update parameters."""
        pass

    @torch.jit.export
    def reset(self):
        """Reset parameters."""
        pass

    @torch.jit.export
    def forward(self, state: Optional[state] = None) -> Action:
        if self._steps % self.solver_frequency == 0 or self.action_sequence is None:
            self.action_sequence = self.solver(state)
        else:
            self.trajectory_optimizer.initialize_actions(state.shape[:-1])

        action = self.action_sequence[self._steps % self.solver_frequency, ..., :]
        self._steps += 1
        return action, torch.zeros(self.dim_action[0], self.dim_action[0])

    def reset(self):
        """Reset trajectory optimizer."""
        self._steps = 0
        self.trajectory_optimizer.reset()
