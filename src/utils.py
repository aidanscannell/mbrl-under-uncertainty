#!/usr/bin/env python3
import logging
import random
import time
from typing import Optional

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np
import pytorch_lightning as pl
import torch
from dm_control import suite
from setuptools.dist import Optional

from src.custom_types import ReplayBuffer
from src.policies import Policy


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(cfg.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    pl.seed_everything(random_seed)


def rollout_agent_and_populate_replay_buffer(
    env: gym.Env,
    policy: Policy,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    # rollout_horizon: Optional[int] = 1,
    rollout_horizon: Optional[int] = None,
) -> ReplayBuffer:
    logger.info(f"Collecting {num_episodes} episodes from env")

    observation, info = env.reset()
    for episode in range(num_episodes):
        terminated, truncated = False, False
        timestep = 0
        while not terminated or truncated:
            if rollout_horizon is not None:
                if timestep >= rollout_horizon:
                    break
            action = policy(observation=observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            replay_buffer.push(
                observation=observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            observation = next_observation

            timestep += 1

        observation, info = env.reset()

    return replay_buffer


def make_env(env_name, seed, action_repeat):
    """
    Make environment for TD-MPC experiments.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = str(env_name).replace("-", "_").split("_", 1)
    domain = dict(cup="ball_in_cup").get(domain, domain)

    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain, task, task_kwargs={"random": seed}, visualize_reward=False
        )
    else:
        import os
        import sys

        sys.path.append("..")
        import custom_dmc_tasks as cdmc

        env = cdmc.make(
            domain, task, task_kwargs={"random": seed}, visualize_reward=False
        )

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = ExtendedTimeStepWrapper(env)
    env = ConcatObsWrapper(env)
    # env = TimeStepToGymWrapper(env, domain, task, action_repeat, 'state')

    return env


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time
