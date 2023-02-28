#!/usr/bin/env python3
import abc

import dm_env

from src.custom_types import Action, ReplayBuffer


class Agent(abc.ABC):
    def act(self, timestep: dm_env.TimeStep) -> Action:
        raise NotImplementedError

    def update(self, timestep: dm_env.TimeStep, action, next_timestep: dm_env.TimeStep):
        raise NotImplementedError


class ModelBasedAgent:
    def __init__(self, model, policy, replay_buffer_capacity: int):
        self.model = model
        self.policy = policy
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    def act(self, timestep: dm_env.TimeStep) -> Action:
        return self.policy(timestep)

    def update(self, timestep: dm_env.TimeStep, action, next_timestep: dm_env.TimeStep):
        # self.replay_buffer.add()
        self.replay_buffer.push(
            observation=timestep.observation,
            action=timestep.action,
            next_observation=next_timestep.observation,
            reward=timestep.reward,
            # discount=timestep.discount,
        )
        self.model.train(self.replay_buffer)
        self.policy.train(self.replay_buffer, self.model)
