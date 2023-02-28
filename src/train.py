#!/usr/bin/env python3
import logging
import random
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchrl
from dm_env import specs
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from setuptools.dist import Optional
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.gym import GymEnv
from tensordict.nn import TensorDictModule
from torch import nn, optim
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import CompositeSpec, TensorDictReplayBuffer
from torchrl.data.postprocs import MultiStep
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
import dmc
import utils
import wandb

from src.policies import Policy, RandomPolicy
from src.utils import (
    make_env,
    rollout_agent_and_populate_replay_buffer,
    set_seed_everywhere,
)
from video import TrainVideoRecorder, VideoRecorder

# def eval(agent, eval_env, global_frame, global_step):
#     step, episode, total_reward = 0, 0, 0
#     eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

#     while eval_until_episode(episode):
#         time_step = eval_env.reset()
#         video_recorder.init(eval_env, enabled=(episode == 0))
#         while not time_step.last():
#             with torch.no_grad(), utils.eval_mode(agent):
#                 action = agent.act(time_step.observation, global_step, eval_mode=True)
#             time_step = eval_env.step(action)
#             video_recorder.record(eval_env)
#             total_reward += time_step.reward
#             step += 1

#         episode += 1
#         video_recorder.save(f"{global_frame}.mp4")


#     wandb.log({"episode_reward": total_reward / episode})
#     wandb.log({"episode_length": step * cfg.action_repeat / episode})
#     wandb.log({"episode": global_episode})
#     wandb.log({"step": global_step})
def make_replay_buffer(buffer_size, device, buffer_scratch_dir, prefetch=3):
    sampler = RandomSampler()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            buffer_size,
            scratch_dir=buffer_scratch_dir,
            device=device,
        ),
        sampler=sampler,
        pin_memory=False,
        prefetch=prefetch,
    )
    return replay_buffer


def make_recorder(actor_model_explore, transform_state_dict):
    base_env = make_env()
    recorder = make_transformed_env(base_env)
    recorder.transform[2].init_stats(3)
    recorder.transform[2].load_state_dict(transform_state_dict)

    recorder_obj = Recorder(
        record_frames=1000,
        frame_skip=frame_skip,
        policy_exploration=actor_model_explore,
        recorder=recorder,
        exploration_mode="mean",
        record_interval=record_interval,
    )
    return recorder_obj


def make_env(cfg, device):
    """Create a base env."""

    hydra.instantiate(cfg.env, device=device)
    env_kwargs = {
        "device": device,
        "frame_skip": frame_skip,
        "from_pixels": from_pixels,
        "pixels_only": from_pixels,
    }
    env = env_library(*env_args, **env_kwargs)
    return env


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    work_dir = Path.cwd()
    timer = utils.Timer()
    try:  # Make experiment reproducible
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Fetching the device that will be used throughout this notebook
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )

    # Initialise WandB run
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=cfg.wandb.tags,
        name=cfg.wandb.run_name,
    )
    # wandb_logger = WandbLogger(name="Adam-32-0.001", project="pytorchlightning")
    log_dir = run.dir

    # Configure environment
    train_env = hydra.utils.instantiate(cfg.env)
    train_env.set_seed(cfg.random_seed)
    print("train_env")
    print(train_env)
    # train_env = dmc.make(
    #     cfg.env_name, cfg.task_name, cfg.frame_stack, cfg.action_repeat, cfg.random_seed
    # )
    # if torch.has_cuda and torch.cuda.device_count():
    #     train_env.to("cuda:0")
    #     train_env.reset()

    dynamic_model = hydra.utils.instantiate(cfg.model)
    print("dynamic_model")
    print(dynamic_model)
    print(type(dynamic_model))
    # Configure agent
    # agent = hydra.utils.instantiate(cfg.agent, env=env)
    # agent = hydra.utils.instantiate(cfg.agent)
    # policy = hydra.utils.instantiate(cfg.policy)

    # Collect initial data set
    # replay_buffer = rollout_agent_and_populate_replay_buffer(
    #     env=train_env,
    #     policy=RandomPolicy(action_spec=train_env.action_spec),
    #     replay_buffer=ReplayBuffer(capacity=cfg.initial_dataset.replay_buffer_capacity),
    #     num_episodes=cfg.initial_dataset.num_episodes,
    # )
    # replay_buffer = ReplayBuffer(capacity=cfg.initial_dataset.replay_buffer_capacity)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(size=cfg.replay_buffer_capacity, device=device),
        sampler=SamplerWithoutReplacement(),
    )
    logger = torchrl.record.loggers.WandbLogger(exp_name=cfg.run_name)
    video_recorder = torchrl.record.VideoRecorder(logger)
    # torchrl.record.VideoRecorder(logger, tag, in_keys, skip: int = 2)

    # Create replay buffer
    # train_env.observation_spec()["name"] = "obs"
    # obs_spec = train_env.observation_spec()
    # obs_spec.name = "obs"
    # obs_sepc=specs.BoundedArray(shape=np.concatenate(
    #         [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
    #                                         dtype=np.uint8,
    #                                         minimum=0,
    #                                         maximum=255,
    #                                         name='observation')
    # print("obs_spec")
    # print(obs_spec)
    # print(train_env.observation_spec())
    # data_specs = (
    #     # obs_spec,
    #     # train_env.observation_spec(),
    #     train_env.action_spec(),
    #     specs.Array((1,), np.float32, "reward"),
    #     specs.Array((1,), np.float32, "discount"),
    # )
    # replay_storage = ReplayBufferStorage(data_specs, work_dir / "buffer")
    # replay_loader = make_replay_loader(
    #     work_dir / "buffer",
    #     cfg.replay_buffer_size,
    #     cfg.batch_size,
    #     cfg.replay_buffer_num_workers,
    #     cfg.save_snapshot,
    #     cfg.nstep,
    #     cfg.discount,
    # )

    # if cfg.save_video:
    #     video_recorder = VideoRecorder(work_dir)
    # if cfg.save_train_video:
    #     train_video_recorder = TrainVideoRecorder(work_dir)

    policy = RandomPolicy(action_spec=train_env.action_spec)
    policy_module = TensorDictModule(
        policy, in_keys=["observation"], out_keys=["action"]
    )
    # logger.info("Training dynamic model")
    # dynamic_model.train(replay_buffer)
    # logger.info("DONE TRAINING MODEL")

    global_step = 0
    for episode in range(cfg.num_episodes):
        # Run the RL training loop
        episode_reward = 0
        # time_step = train_env.reset()
        tensordict = train_env.reset()

        # if cfg.save_train_video:
        #     train_video_recorder.init(time_step.observation)
        tensordict_rollout = train_env.rollout(
            max_steps=cfg.max_steps_per_episode, policy=policy_module
        )
        print("tensordict_rollout")
        print(tensordict_rollout)

        for episode_step in range(cfg.max_steps_per_episode):
            # while train_until_step(self.global_step):
            # while not time_step.last():
            # if time_step.last():
            if done:
                break

            # Sample action
            with torch.no_grad():
                action = policy(time_step.observation)
                # action = action.cpu().numpy()

            # Take env step
            next_obs, reward, done, info = train_env.step(action)
            episode_reward += reward
            # replay_storage.add(time_step)
            # print("time step")
            # print(time_step)
            replay_buffer.push(
                observation=obs,
                action=action,
                next_observation=next_obs,
                reward=reward,
                terminated=False,
                truncated=False,
                # discount=time_step.discount,
            )
            obs = next_obs

            # if cfg.save_train_video:
            #     train_video_recorder.record(time_step.observation)
            global_step += 1

            # if global_step % cfg.eval_every == 0:
            if cfg.use_wandb:
                # log stats
                elapsed_time, total_time = timer.reset()

                wandb.log({"episode_reward": episode_reward})
                # wandb.log({"episode_length": episode_step})
                wandb.log({"episode": episode})
                wandb.log({"step": global_step})

        # if cfg.save_train_video:
        #     train_video_recorder.save(f"{episode}.mp4")

        # Log stats
        elapsed_time, total_time = timer.reset()
        # wandb.log({"fps", episode_frame / elapsed_time})
        wandb.log({"total_time", total_time})
        wandb.log({"episode_reward", episode_reward})
        wandb.log({"episode_length", episode_step})
        wandb.log({"episode", episode})
        wandb.log({"buffer_size", len(replay_storage)})
        wandb.log({"step", global_step})

        # try to save snapshot
        # if self.cfg.save_snapshot:
        #     self.save_snapshot()

        # Train the agent ⚡
        logger.info("Training dynamic model...")
        # dynamic_model.train(replay_storage)
        logger.info("Done training dynamic model")
        # policy.train(replay_buffer, dynamic_model)


if __name__ == "__main__":
    train()  # pyright: ignore
