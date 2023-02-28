#!/usr/bin/env python3
"""
The goal of RL is to maximize the discounted sum of rewards in expectation over the transition noise ε_t.
That is, to maximize
    J(f, π) = E_{ε_{0:∞}} Σ_{t=0}^{∞} γ^{t} r(s_t, a_t)
with transition dynamics given by
    s_{t+1} = f(s_t, a_t) + ε_t

In model-based RL, we often obtain a posterior distribution over the dynamics model p(f | D).
Here we define multiple objectives which utilise p(f | D) in different ways.
"""
from typing import Callable, Optional

import torch
from torch import vmap
from torchtyping.tensor_type import TensorType

from ..custom_types import Action, ActionTrajectory, Observation
from ..models.models import DynamicsModel
from ..policies.base import Policy

ObjectiveFn = Callable[[ActionTrajectory], TensorType["1"]]


class Objective:
    def __init__(self, start_state):
        self.start_state = start_state

    def __call__(self, start_state, policy: Policy):
        pass


def greedy_objective(
    model: DynamicsModel,
    policy: Policy,
    start_state,
    state_prop_method,
    horizon: int,
    reward_fn: Optional[Callable[[Observation, Action], TensorType["1"]]] = None,
    value_fn: Optional[Callable[[Observation], TensorType["1"]]] = None,
):
    r"""Greedy objective given by,

        J_{greedy}(f, π) = E_{p(f | D)}[J(f, π)]

    Given the RL objective:
        J(f, π) = E_{ε_{0:∞}} Σ_{t=0}^{∞} r(s_t, a_t)
    s.t. transition dynamics
        s_{t+1} = f(s_t, a_t) + ε_t

    This objective is used in PILCO/PETS/GP-MPC to name a few
    """

    # def single_predict_output(state):
    #     action = policy(state)
    #     prediction = model.predict(observation=state, action=action)
    #     return prediction.output_dist.mean(), prediction.output_dist.variance()
    #     # return prediction.output_dist.mean(), prediction.output_dist.variance()

    # def single_predict_latent(state):
    #     action = policy(state)
    #     prediction = model.predict(observation=state, action=action)
    #     return prediction.latent_dist.mean(), prediction.output_dist.variance()

    # output_predict_fn = vmap(single_predict_output)
    # latent_predict_fn = vmap(single_predict_latent)

    # prediction = model.propagate(state_dist=state_dist, action=action)

    # def trajectory_sampling(state):
    #     action = policy(state)
    #     x = torch.concat([state, action], -1)

    # def distribution_sampling(state):
    #     action = policy(state)
    #     x = torch.concat([state, action], -1)
    #     # prediction = model.predict(observation=state, action=action)
    #     y_means, y_vars = [], []
    #     # TODO make this run in parallel
    #     for network in model.networks:
    #         y_mean, y_var = network.forward_mean_var(x)
    #         y_means.append(y_mean)
    #         y_vars.append(y_var)

    #     y_mean = torch.mean(y_means)
    #     y_var = torch.var(y_vars)
    #     next_state_dist = td.Normal(loc=y_mean, scale=torch.sqrt(y_var))
    def single_predict_output(state):
        action = policy(state)
        prediction = model.predict(observation=state, action=action)
        return prediction.output_dist.mean(), prediction.output_dist.variance()

    predictions = []
    num_samples = 20
    # state = start_state
    # next_state_means, next_state_vars = single_predict(start_state)
    action = policy(start_state)
    prediction = model.predict(observation=start_state, action=action)
    next_state_samples = prediction.output_dist.sample(torch.Size([num_samples]))
    print("next_state_samples.shape")
    print(next_state_samples.shape)

    for _ in range(1, horizon):
        next_state_means, next_state_vars = predict_fn(next_state_samples)
        # action = policy(state)
        # prediction = model.predict(observation=state, action=action)
        # prediction.output_dist.sample()
        # predictions.append(model.predict(observation=observation, action=action))

    value = reward_fn(states)
    if value_fn is not None:
        value = value + value_fn(states[-1, :])


class Greedy:
    def __init__(self, start_state):
        self.start_state = start_state

    def __call__(self, start_state, policy: Policy):
        pass
