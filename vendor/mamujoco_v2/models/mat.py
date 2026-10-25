"""Mat."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def init_(layer, activate=False, gain=0.01):
    """Init."""
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu") if activate else gain)
    nn.init.zeros_(layer.bias)
    return layer


class SelfAttention(nn.Module):
    """Selfattention."""

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

        x = self.norm2(encoded + self.cross_attention(encoded, x, x))
        return self.norm3(x + self.mlp(x))




class MATActor(nn.Module):
    """Matactor."""

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

        self.action_embedding = nn.Sequential(init_(nn.Linear(self.n_actions + 1, width), activate=True), nn.GELU())
        self.mu_head = nn.Sequential(
            init_(nn.Linear(width, width), activate=True), nn.GELU(), nn.LayerNorm(width),
            init_(nn.Linear(width, self.n_actions)),
        )

        sigma = config["sigma"]
        if sigma["parameterization"] == "sigmoid":
            probability = (sigma["initial"] - sigma["minimum"]) / (sigma["maximum"] - sigma["minimum"])
            raw = math.log(probability / (1.0 - probability))
        else:
            raw = math.log(sigma["initial"] - sigma["floor"])
        self.raw_sigma = nn.Parameter(torch.full((self.n_agents, self.n_actions), raw))



    def init_hidden(self, batch_size):
        return self.fc1.weight.new_zeros(batch_size, self.n_agents, self.hidden_dim)

    def encode(self, inputs, hidden):
        """Encode."""
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
        """Encode sequence."""
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
        """Decode."""
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



    def sigma(self):
        """sigmoid: σ ∈ [minimum, maximum]；exp: σ = floor + exp(raw)。"""
        settings = self.sigma_config
        if settings["parameterization"] == "exp":
            return settings["floor"] + self.raw_sigma.exp()
        return settings["minimum"] + (settings["maximum"] - settings["minimum"]) * self.raw_sigma.sigmoid()

    def log_prob(self, raw_action, mean):
        """Log prob."""
        sigma = self.sigma()
        per_dimension = -0.5 * ((raw_action - mean) / sigma).square() - sigma.log() - 0.5 * math.log(2.0 * math.pi)
        return (per_dimension * self.action_mask).sum(dim=-1)

    def entropy(self):
        per_dimension = 0.5 * math.log(2.0 * math.pi * math.e) + self.sigma().log()
        return (per_dimension * self.action_mask).sum(dim=-1)

    @torch.no_grad()
    def sample_actions(self, inputs, hidden, latent, sample_noise=True):
        """Sample actions."""
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
        """Evaluate actions."""
        features = self.encode_sequence(inputs, bptt_steps).reshape(-1, self.n_agents, self.hidden_dim)
        raw = raw_actions[indices]
        mean = self.decode(self.encode_agents(features[indices]), self.shift(raw.sigmoid() * self.action_mask))
        return {"log_probs": self.log_prob(raw, mean), "entropy": self.entropy()}
