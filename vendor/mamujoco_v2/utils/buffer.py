"""Rollout tensors, validity masking and generalised advantage estimation."""

from dataclasses import dataclass, fields

import torch


@dataclass
class RolloutBatch:
    """Rolloutbatch."""

    observations: torch.Tensor     # [B,T+1,N,O]
    states: torch.Tensor           # [B,T+1,S]
    inputs: torch.Tensor
    actions: torch.Tensor
    raw_actions: torch.Tensor
    means: torch.Tensor            # [B,T,N,A]
    noise: torch.Tensor            # [B,T,N,A]
    latent: torch.Tensor
    log_probs: torch.Tensor        # [B,T,N]
    rewards: torch.Tensor          # [B,T,1]
    terminated: torch.Tensor
    truncated: torch.Tensor
    filled: torch.Tensor           # [B,T,1]

    @property
    def env_steps(self):
        return int(self.filled.sum().item())

    @classmethod
    def field_names(cls):
        return [field.name for field in fields(cls)]

    @classmethod
    def stack(cls, rows):
        """Stack."""
        return cls(**{name: torch.stack(rows[name], dim=1) for name in cls.field_names()})

    @classmethod
    def concatenate(cls, batches, limit=None):
        """Concatenate."""
        values = {}
        for name in cls.field_names():
            parts = [getattr(batch, name) for batch in batches]
            length = max(part.shape[1] for part in parts)
            parts = [torch.cat((part, part.new_zeros(part.shape[0], length - part.shape[1], *part.shape[2:])), dim=1)
                     if part.shape[1] < length else part for part in parts]
            tensor = torch.cat(parts, dim=0)
            values[name] = tensor if limit is None else tensor[:limit]
        return cls(**values)


def compute_gae(rewards, values, terminated, filled, gamma, gae_lambda):
    """Compute gae."""
    advantages = torch.zeros_like(values[:, :-1])
    accumulator = torch.zeros_like(values[:, 0])
    for time in reversed(range(rewards.shape[1])):
        valid = filled[:, time].to(values.dtype)
        continuation = (~terminated[:, time]).to(values.dtype)
        delta = rewards[:, time] + gamma * continuation * values[:, time + 1] - values[:, time]
        accumulator = (delta + gamma * gae_lambda * continuation * accumulator) * valid
        advantages[:, time] = accumulator
    return advantages, advantages + values[:, :-1]


def normalize_advantages(advantages, valid):
    """Normalize advantages."""
    selected = advantages[valid]
    advantages = advantages.clone()
    advantages[valid] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    return advantages
