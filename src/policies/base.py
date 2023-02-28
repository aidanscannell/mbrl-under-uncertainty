#!/usr/bin/env python3
from abc import ABCMeta
from typing import Optional

import numpy as np
import torch
import torch.jit
import torch.nn as nn
from ..custom_types import Action, State, ReplayBuffer


class Policy(nn.Module, metaclass=ABCMeta):
    """Interface for RL policies"""

    def __init__(self):
        # def __init__(self, state_dim: int, action_dim: int, *args, **kwargs):
        super().__init__()
        # self.state = state_dim
        # self.action = action_dim

    @torch.jit.export
    def __call__(self, state: Optional[State] = None, *args, **kwargs) -> Action:
        return self.act(state=state)

    @torch.jit.export
    def act(self, state: Optional[State] = None) -> Action:
        raise NotImplementedError

    @torch.jit.export
    def update(self, replay_buffer: ReplayBuffer, *args, **kwargs):
        """Update parameters."""
        pass

    @torch.jit.export
    def reset(self):
        """Reset parameters."""
        pass
