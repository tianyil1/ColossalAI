import os
import random

import numpy as np
import torch
import torch.nn as nn
from . import extend_distributed as ext_dist
from coati.models.base import Actor
from coati.models.lora import LoraLinear
from coati.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .base import Strategy
from .naive import NaiveStrategy
from .sampler import DistributedSampler


class IntelDDPStrategy(NaiveStrategy):
    """
        Strategy for distributed training using torch.distributed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        super().__init__()

    def setup_distributed(self) -> None:
        self.set_seed(self.seed)
        ext_dist.init_distributed()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def setup_model(self, model: nn.Module) -> nn.Module:
        if ext_dist.my_size > 1:
            return ext_dist.DDP(model)

    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        # DDP only mode, replay buffers on each rank are different.
        # sampler = DistributedSampler(replay_buffer,
        #                              num_replicas=dist.get_world_size(),
        #                              rank=dist.get_rank(),
        #                              shuffle=True,
        #                              seed=self.seed,
        #                              drop_last=True)
        return DataLoader(
            replay_buffer,
            batch_size=replay_buffer.sample_batch_size,
        #   sampler=sampler,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
            collate_fn=replay_buffer.collate_fn)

    @staticmethod
    def _unwrap_actor(actor: Actor) -> nn.Module:
        model: ext_dist.DDP = Strategy._unwrap_actor(actor)
        return model.module

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.merge_weights = True
                module.eval()

        if only_rank0 and ext_dist.dist.get_rank() != 0:
            return
        model = model.model.module
        state_dict = model.state_dict()
        torch.save(state_dict, path)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        if only_rank0 and ext_dist.dist.get_rank() != 0:
            return
        super().save_optimizer(optimizer, path, only_rank0)

    def setup_sampler(self, dataset) -> DistributedSampler:
        return DistributedSampler(dataset, ext_dist.dist.get_world_size(), ext_dist.dist.get_rank())
