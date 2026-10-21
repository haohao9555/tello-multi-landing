"""所有算法共用的 centralized critic：网络结构、target 网络、优化器、GAE 目标和回归更新都在这里。

MAPPO / HAPPO / MAT 的 critic 完全相同（每个 run 各自一份权重），算法之间只有 actor 不同。
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from utils.buffer import compute_gae


class CentralCritic(nn.Module):
    """归一化 state + agent one-hot -> 每个 agent 的 V。"""

    def __init__(self, state_dim, n_agents, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.fc1 = nn.Linear(state_dim + n_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        leading = state.shape[:-1]
        states = state.unsqueeze(-2).expand(*leading, self.n_agents, state.shape[-1])
        identities = torch.eye(self.n_agents, device=state.device, dtype=state.dtype)
        identities = identities.expand(*leading, self.n_agents, self.n_agents)
        features = torch.cat((states, identities), dim=-1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(features))))).squeeze(-1)


class SharedCritic:
    """critic 网络 + target 网络 + Adam；提供 GAE 目标、全 batch 回归一步、target 软更新。"""

    def __init__(self, env_info, config, normalizers, device):
        self.config = config
        self.normalizers = normalizers
        self.network = CentralCritic(env_info["state_dim"], env_info["n_agents"], config["network"]["hidden_dim"]).to(device)
        self.target = copy.deepcopy(self.network).requires_grad_(False)
        training = config["training"]
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=training["critic_lr"], eps=training["optimizer_eps"])

    @torch.no_grad()
    def gae(self, batch, rewards, use_target):
        """(advantages, returns)，均为 [B,T,N]；returns 已按 running 统计归一化（若开启）。
        use_target=False 用当前 critic（gae_value=online），True 用 target critic（gae_value=target）。"""
        network = self.target if use_target else self.network
        values = network(self.normalizers.normalize_states(batch.states))
        normalize_returns = self.config["normalization"]["returns"]
        if normalize_returns:
            values = self.normalizers.returns.denormalize(values)
        training = self.config["training"]
        advantages, returns = compute_gae(
            rewards, values, batch.terminated, batch.filled, training["gamma"], training["gae_lambda"]
        )
        if normalize_returns:
            self.normalizers.returns.update(returns[batch.filled.squeeze(-1).bool()])
            returns = self.normalizers.returns.normalize(returns)
        return advantages, returns

    def update(self, batch, returns, valid):
        """全 batch MSE 回归一步；返回 (loss, grad_norm)。"""
        chunks = self.config["training"].get("critic_minibatches", 1)
        if chunks > 1:
            states = self.normalizers.normalize_states(batch.states[:, :-1])[valid]
            targets = returns[valid]
            for indices in torch.randperm(states.shape[0], device=states.device).chunk(chunks):
                loss = (self.network(states[indices]) - targets[indices]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("critic loss 非有限；请检查学习率和归一化配置")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config["training"]["grad_norm_clip"], error_if_nonfinite=True
                )
                self.optimizer.step()
            return loss.detach(), grad
        predicted = self.network(self.normalizers.normalize_states(batch.states[:, :-1]))
        loss = (predicted[valid] - returns[valid]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("critic loss 非有限；请检查学习率和归一化配置")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(
            self.network.parameters(), self.config["training"]["grad_norm_clip"], error_if_nonfinite=True
        )
        self.optimizer.step()
        return loss.detach(), grad

    @torch.no_grad()
    def soft_update(self):
        for target, online in zip(self.target.parameters(), self.network.parameters()):
            target.lerp_(online, self.config["training"]["target_tau"])

    def train(self, mode=True):
        self.network.train(mode)

    def state_dict(self):
        return {"critic": self.network.state_dict(), "target_critic": self.target.state_dict(),
                "critic_optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state, resume):
        self.network.load_state_dict(state["critic"])
        self.target.load_state_dict(state["target_critic"])
        if resume:
            self.optimizer.load_state_dict(state["critic_optimizer"])
