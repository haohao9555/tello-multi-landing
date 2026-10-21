"""Rollout 数据：一批定长 episode 的张量容器，以及 GAE。"""

from dataclasses import dataclass, fields

import torch


@dataclass
class RolloutBatch:
    """B 条 episode × T 步；filled=False 的位置是 episode 提前结束后的 padding。"""

    observations: torch.Tensor     # [B,T+1,N,O]
    states: torch.Tensor           # [B,T+1,S]
    inputs: torch.Tensor           # [B,T,N,F]  actor 输入（归一化 obs + 可选 agent ID）
    actions: torch.Tensor          # [B,T,N,A]  环境动作 sigmoid(raw)
    raw_actions: torch.Tensor      # [B,T,N,A]  高斯样本 μ + σn
    means: torch.Tensor            # [B,T,N,A]
    noise: torch.Tensor            # [B,T,N,A]
    latent: torch.Tensor           # [B,T,N,A]  latent 接口项（MAPPO / HAPPO / MAT 不使用）
    log_probs: torch.Tensor        # [B,T,N]
    rewards: torch.Tensor          # [B,T,1]
    terminated: torch.Tensor       # [B,T,1]  只有真实终止为 True
    truncated: torch.Tensor        # [B,T,1]  到达时间上限为 True
    filled: torch.Tensor           # [B,T,1]

    @property
    def env_steps(self):
        return int(self.filled.sum().item())

    @classmethod
    def field_names(cls):
        return [field.name for field in fields(cls)]

    @classmethod
    def stack(cls, rows):
        """rows: {字段名: 按时间排列的张量列表} -> 在时间维堆叠。"""
        return cls(**{name: torch.stack(rows[name], dim=1) for name in cls.field_names()})

    @classmethod
    def concatenate(cls, batches, limit=None):
        """在 episode 维拼接；各批时间长度可能不同（全部 episode 结束即停止采样），短的在时间维补零（filled=False）。"""
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
    """values: [B,T+1,N]；只有真实终止切断 bootstrap，时间截断仍用下一状态的 V。"""
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
    """只用有效位置的均值/标准差做标准化。"""
    selected = advantages[valid]
    advantages = advantages.clone()
    advantages[valid] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    return advantages
