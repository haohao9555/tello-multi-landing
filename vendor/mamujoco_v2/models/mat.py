"""MAT actor（Wen et al. NeurIPS 2022）：agent 间 Transformer encoder + 按固定 agent 顺序自回归的 decoder。

- encoder：对 N 个 agent 的观测编码做 self-attention；
- decoder：输入右移一位的动作序列（首位为起始标记），先做 causal self-attention，
  再以 encoder 表示为 query、动作序列为 key/value 做 causal cross-attention；
- 采样：逐 agent 自回归，agent i 条件于 agent 1..i-1 已采样的环境动作；
- PPO 重算：用已存动作 teacher forcing，一次并行得到逐 agent 条件 log p。
简化：critic 用公共 centralized critic（models/common_critic.py），不用 MAT 原版的 encoder 价值头。
初始化同官方：attention 的 q/k/v/输出 N(0, 0.02²)、bias 0；其余线性层正交初始化，
带激活的层 gain=ReLU 增益，末层 gain=0.01。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def init_(layer, activate=False, gain=0.01):
    """官方 init_：正交权重、零 bias；带激活时 gain 取 ReLU 增益。"""
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu") if activate else gain)
    nn.init.zeros_(layer.bias)
    return layer


class SelfAttention(nn.Module):
    """多头注意力；causal=True 时 agent i 只看 agent ≤ i。"""

    def __init__(self, width, heads, causal):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, query, key, value):
        """query/key/value [M,N,H] -> [M,N,H]。"""
        batch, n_agents, width = query.shape
        split = lambda x: x.reshape(batch, n_agents, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(key)), split(self.value(value))
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        if self.causal:
            future = torch.ones(n_agents, n_agents, dtype=torch.bool, device=query.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        mixed = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, n_agents, width)
        return self.output(mixed)


class EncodeBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = SelfAttention(width, heads, causal=False)
        self.mlp = nn.Sequential(init_(nn.Linear(width, width), activate=True), nn.GELU(),
                                 init_(nn.Linear(width, width)))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x))
        return self.norm2(x + self.mlp(x))


class DecodeBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attention = SelfAttention(width, heads, causal=True)
        self.cross_attention = SelfAttention(width, heads, causal=True)
        self.mlp = nn.Sequential(init_(nn.Linear(width, width), activate=True), nn.GELU(),
                                 init_(nn.Linear(width, width)))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, x, encoded):
        x = self.norm1(x + self.self_attention(x, x, x))
        # 与官方 DecodeBlock 一致：query 为 encoder 表示，key/value 为动作序列。
        x = self.norm2(encoded + self.cross_attention(encoded, x, x))
        return self.norm3(x + self.mlp(x))




class MATActor(nn.Module):
    """观测编码（MLP 或 GRU）-> Transformer encoder -> 自回归 decoder -> Gaussian 均值。
    约定：B=batch，N=agent 数，A=动作维（含 padding，由 action_mask 屏蔽），H=隐藏维。"""

    def __init__(self, input_dim, env_info, config):
        super().__init__()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.hidden_dim = config["network"]["hidden_dim"]
        self.use_gru = config["network"]["use_gru"]
        self.sigma_config = config["sigma"]
        self.register_buffer("action_mask", torch.as_tensor(env_info["action_mask"], dtype=torch.float32))
        width, heads, layers = self.hidden_dim, config["attention_heads"], config["layers"]

        self.fc1 = nn.Linear(input_dim, width)
        self.rnn = nn.GRU(width, width) if self.use_gru else nn.Linear(width, width)
        self.encoder = nn.ModuleList(EncodeBlock(width, heads) for _ in range(layers))
        self.decoder = nn.ModuleList(DecodeBlock(width, heads) for _ in range(layers))
        # 动作 token：[起始标记, 上一个 agent 的环境动作]。
        self.action_embedding = nn.Sequential(init_(nn.Linear(self.n_actions + 1, width), activate=True), nn.GELU())
        self.mu_head = nn.Sequential(
            init_(nn.Linear(width, width), activate=True), nn.GELU(), nn.LayerNorm(width),
            init_(nn.Linear(width, self.n_actions)),
        )

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

    # ---------- Transformer ----------

    def encode_agents(self, features):
        encoded = features.reshape(-1, self.n_agents, self.hidden_dim)
        for block in self.encoder:
            encoded = block(encoded)
        return encoded

    def decode(self, encoded, previous_actions):
        """previous_actions [M,N,A]：agent i 位置放 agent i-1 的环境动作；返回均值 [M,N,A]。"""
        start = previous_actions.new_zeros(*previous_actions.shape[:-1], 1)
        start[:, 0] = 1
        tokens = self.action_embedding(torch.cat((start, previous_actions), dim=-1))
        for block in self.decoder:
            tokens = block(tokens, encoded)
        return self.mu_head(tokens) * self.action_mask

    @staticmethod
    def shift(actions):
        shifted = torch.zeros_like(actions)
        shifted[:, 1:] = actions[:, :-1]
        return shifted

    # ---------- 高斯 ----------

    def sigma(self):
        """sigmoid：σ ∈ [minimum, maximum]；exp：σ = floor + exp(raw)。"""
        settings = self.sigma_config
        if settings["parameterization"] == "exp":
            return settings["floor"] + self.raw_sigma.exp()
        return settings["minimum"] + (settings["maximum"] - settings["minimum"]) * self.raw_sigma.sigmoid()

    def log_prob(self, raw_action, mean):
        """逐 agent 条件高斯 log p（条件于前序 agent 的动作），只对有效动作维求和。"""
        sigma = self.sigma()
        per_dimension = -0.5 * ((raw_action - mean) / sigma).square() - sigma.log() - 0.5 * math.log(2.0 * math.pi)
        return (per_dimension * self.action_mask).sum(dim=-1)

    def entropy(self):
        per_dimension = 0.5 * math.log(2.0 * math.pi * math.e) + self.sigma().log()
        return (per_dimension * self.action_mask).sum(dim=-1)

    @torch.no_grad()
    def sample_actions(self, inputs, hidden, latent, sample_noise=True):
        """逐 agent 自回归采样；输入 decoder 的是 sigmoid 后的环境动作。latent 仅为接口占位。"""
        features = self.encode(inputs, hidden)
        encoded = self.encode_agents(features)
        sigma = self.sigma()
        previous = features.new_zeros(*features.shape[:-1], self.n_actions)
        means, raws = torch.zeros_like(previous), torch.zeros_like(previous)
        for agent in range(self.n_agents):
            mean = self.decode(encoded, previous)[:, agent]
            noise = torch.randn_like(mean) * sigma[agent] if sample_noise else torch.zeros_like(mean)
            means[:, agent] = mean
            raws[:, agent] = (mean + noise) * self.action_mask[agent]
            if agent + 1 < self.n_agents:
                previous[:, agent + 1] = raws[:, agent].sigmoid() * self.action_mask[agent]
        return {
            "actions": raws.sigmoid() * self.action_mask,
            "raw_actions": raws,
            "means": means,
            "noise": raws - means,
            "latent": latent * self.action_mask,
            "log_probs": self.log_prob(raws, means),
        }, features

    def evaluate_actions(self, inputs, raw_actions, indices, bptt_steps=None):
        """teacher forcing：用已存动作一次并行得到逐 agent 条件均值。"""
        features = self.encode_sequence(inputs, bptt_steps).reshape(-1, self.n_agents, self.hidden_dim)
        raw = raw_actions[indices]
        mean = self.decode(self.encode_agents(features[indices]), self.shift(raw.sigmoid() * self.action_mask))
        return {"log_probs": self.log_prob(raw, mean), "entropy": self.entropy()}
