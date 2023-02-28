#!/usr/bin/env python3
from abc import ABCMeta
from typing import Optional
import dm_env

import gymnasium as gym
import numpy as np
import torch
import torch.jit
import torch.nn as nn

from ..custom_types import Action, State, ReplayBuffer
from .base import Policy


class RandomPolicy(Policy):
    def __init__(self, action_spec: dm_env.specs.BoundedArray):
        self.action_spec = action_spec

    def act(self, state: Optional[State] = None) -> Action:
        print(self.action_spec)
        return np.random.uniform(
            self.action_spec.space.minimum,
            self.action_spec.space.maximum,
            size=self.action_spec.shape,
            # self.action_spec().minimum,
            # self.action_spec().maximum,
            # size=self.action_spec().shape,
        )
