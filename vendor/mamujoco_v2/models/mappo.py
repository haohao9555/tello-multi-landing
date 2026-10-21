"""MAPPO actor：参数在 agent 间共享的 Gaussian 策略。

h = enc(o)（MLP 或 GRU），μ = W2·relu(W1·h)，raw ~ N(μ, σ²)，环境动作 a = sigmoid(raw)。
约定：B=batch，N=agent 数，A=动作维（含 padding，由 action_mask 屏蔽），H=隐藏维。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MAPPOActor(nn.Module):
    def __init__(self, input_dim, env_info, config):
        super().__init__()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.hidden_dim = config["network"]["hidden_dim"]
        self.use_gru = config["network"]["use_gru"]
        self.sigma_config = config["sigma"]
        self.action_squash = config.get("action_squash", "sigmoid")
        self.register_buffer("action_mask", torch.as_tensor(env_info["action_mask"], dtype=torch.float32))
        width = self.hidden_dim
        self.fc1 = nn.Linear(input_dim, width)
        self.rnn = nn.GRU(width, width) if self.use_gru else nn.Linear(width, width)
        self.mu_fc1 = nn.Linear(width, width)
        self.mu_fc2 = nn.Linear(width, self.n_actions)
        sigma = config["sigma"]                 # 把 σ 的初值反解成可学习参数 raw 的初值
        if sigma["parameterization"] == "sigmoid":
            probability = (sigma["initial"] - sigma["minimum"]) / (sigma["maximum"] - sigma["minimum"])
            raw = math.log(probability / (1.0 - probability))
        else:
            raw = math.log(sigma["initial"] - sigma["floor"])
        self.raw_sigma = nn.Parameter(torch.full((self.n_agents, self.n_actions), raw))

    # ---------- 观测编码：采样时单步，训练时整段 ----------

    def init_hidden(self, batch_size):
        return self.fc1.weight.new_zeros(batch_size, self.n_agents, self.hidden_dim)

    def encode(self, inputs, hidden):
        """inputs [B,N,F] -> features [B,N,H]；GRU 时 features 即下一步 hidden。"""
        batch_size = inputs.shape[0]
        features = F.relu(self.fc1(inputs))
        if not self.use_gru:
            return F.relu(self.rnn(features))
        _, next_hidden = self.rnn(
            features.reshape(1, batch_size * self.n_agents, self.hidden_dim),
            hidden.reshape(1, batch_size * self.n_agents, self.hidden_dim).contiguous(),
        )
        return next_hidden.reshape(batch_size, self.n_agents, self.hidden_dim)

    def encode_sequence(self, inputs, bptt_steps=None):
        """inputs [B,T,N,F] -> [B,T,N,H]；bptt_steps 为 GRU 梯度截断长度（None=整局）。"""
        batch_size, length, n_agents, _ = inputs.shape
        features = F.relu(self.fc1(inputs))
        if not self.use_gru:
            return F.relu(self.rnn(features))
        sequence = features.permute(1, 0, 2, 3).reshape(length, batch_size * n_agents, self.hidden_dim)
        hidden, parts = None, []
        chunk_length = bptt_steps or length
        for start in range(0, length, chunk_length):
            if hidden is not None:
                hidden = hidden.detach()
            outputs, hidden = self.rnn(sequence[start : start + chunk_length], hidden)
            parts.append(outputs)
        outputs = torch.cat(parts, dim=0)
        return outputs.reshape(length, batch_size, n_agents, self.hidden_dim).permute(1, 0, 2, 3)

    # ---------- 均值与高斯 ----------

    def mean(self, features):
        tokens = F.relu(self.mu_fc1(features))
        return self.mu_fc2(tokens) * self.action_mask

    def sigma(self):
        """sigmoid：σ ∈ [minimum, maximum]；exp：σ = floor + exp(raw)。"""
        settings = self.sigma_config
        if settings["parameterization"] == "exp":
            return settings["floor"] + self.raw_sigma.exp()
        return settings["minimum"] + (settings["maximum"] - settings["minimum"]) * self.raw_sigma.sigmoid()

    def log_prob(self, raw_action, mean):
        sigma = self.sigma()
        per_dimension = -0.5 * ((raw_action - mean) / sigma).square() - sigma.log() - 0.5 * math.log(2.0 * math.pi)
        return (per_dimension * self.action_mask).sum(dim=-1)

    def entropy(self):
        per_dimension = 0.5 * math.log(2.0 * math.pi * math.e) + self.sigma().log()
        return (per_dimension * self.action_mask).sum(dim=-1)

    @torch.no_grad()
    def sample_actions(self, inputs, hidden, latent, sample_noise=True):
        """单步采样；latent 只为与其他算法接口一致而保存。返回 (动作字典, 下一步 hidden)。"""
        features = self.encode(inputs, hidden)
        mean = self.mean(features)
        noise = (torch.randn_like(mean) * self.sigma() if sample_noise else torch.zeros_like(mean)) * self.action_mask
        raw_action = (mean + noise) * self.action_mask
        return {
            "actions": (raw_action.sigmoid() if self.action_squash == "sigmoid" else (0.5 + 0.25 * raw_action).clamp(0.0, 1.0))
            * self.action_mask,
            "raw_actions": raw_action,
            "means": mean,
            "noise": noise,
            "latent": latent * self.action_mask,
            "log_probs": self.log_prob(raw_action, mean),
        }, features

    def evaluate_actions(self, inputs, raw_actions, indices, bptt_steps=None):
        """PPO 重算：inputs [B,T,N,F] 整段编码，取展平后 indices 位置的 log p。"""
        features = self.encode_sequence(inputs, bptt_steps).reshape(-1, self.n_agents, self.hidden_dim)
        mean = self.mean(features[indices])
        return {"log_probs": self.log_prob(raw_actions[indices], mean), "entropy": self.entropy()}

