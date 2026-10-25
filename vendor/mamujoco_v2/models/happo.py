"""Happo."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AgentActor(nn.Module):
    """Agentactor."""

    def __init__(self, input_dim, action_mask, config):
        super().__init__()
        self.n_actions = action_mask.shape[-1]
        self.hidden_dim = config["network"]["hidden_dim"]
        self.use_gru = config["network"]["use_gru"]
        self.sigma_config = config["sigma"]
        self.register_buffer("action_mask", torch.as_tensor(action_mask, dtype=torch.float32).reshape(1, -1))
        width = self.hidden_dim
        self.fc1 = nn.Linear(input_dim, width)
        self.rnn = nn.GRU(width, width) if self.use_gru else nn.Linear(width, width)
        self.mu_fc1 = nn.Linear(width, width)
        self.mu_fc2 = nn.Linear(width, self.n_actions)

        sigma = config["sigma"]
        if sigma["parameterization"] == "sigmoid":
            probability = (sigma["initial"] - sigma["minimum"]) / (sigma["maximum"] - sigma["minimum"])
            raw = math.log(probability / (1.0 - probability))
        else:
            raw = math.log(sigma["initial"] - sigma["floor"])
        self.raw_sigma = nn.Parameter(torch.full((1, self.n_actions), raw))

    def encode(self, inputs, hidden):
        """inputs [B,1,F] -> [B,1,H]。"""
        batch_size = inputs.shape[0]
        features = F.relu(self.fc1(inputs))
        if not self.use_gru:
            return F.relu(self.rnn(features))
        _, next_hidden = self.rnn(
            features.reshape(1, batch_size, self.hidden_dim), hidden.reshape(1, batch_size, self.hidden_dim).contiguous()
        )
        return next_hidden.reshape(batch_size, 1, self.hidden_dim)

    def encode_sequence(self, inputs, bptt_steps=None):
        """Encode sequence."""
        batch_size, length, _, _ = inputs.shape
        features = F.relu(self.fc1(inputs))
        if not self.use_gru:
            return F.relu(self.rnn(features))
        sequence = features.permute(1, 0, 2, 3).reshape(length, batch_size, self.hidden_dim)
        hidden, parts = None, []
        chunk_length = bptt_steps or length
        for start in range(0, length, chunk_length):
            if hidden is not None:
                hidden = hidden.detach()
            outputs, hidden = self.rnn(sequence[start : start + chunk_length], hidden)
            parts.append(outputs)
        outputs = torch.cat(parts, dim=0)
        return outputs.reshape(length, batch_size, 1, self.hidden_dim).permute(1, 0, 2, 3)

    def mean(self, features):
        return self.mu_fc2(F.relu(self.mu_fc1(features))) * self.action_mask

    def sigma(self):
        """sigmoid: σ ∈ [minimum, maximum]；exp: σ = floor + exp(raw)。"""
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


class HAPPOActor(nn.Module):
    def __init__(self, input_dim, env_info, config):
        super().__init__()
        self.n_agents = env_info["n_agents"]
        self.hidden_dim = config["network"]["hidden_dim"]
        self.register_buffer("action_mask", torch.as_tensor(env_info["action_mask"], dtype=torch.float32))
        self.agents = nn.ModuleList(
            AgentActor(input_dim, env_info["action_mask"][i], config) for i in range(self.n_agents)
        )

    def init_hidden(self, batch_size):
        return self.action_mask.new_zeros(batch_size, self.n_agents, self.hidden_dim)

    def sigma(self):
        return torch.cat([agent.sigma() for agent in self.agents], dim=0)

    @torch.no_grad()
    def sample_actions(self, inputs, hidden, latent, sample_noise=True):
        """Sample actions."""
        samples, features = [], []
        for i, agent in enumerate(self.agents):
            feature = agent.encode(inputs[:, i : i + 1], hidden[:, i : i + 1])
            mean = agent.mean(feature)
            noise = (torch.randn_like(mean) * agent.sigma() if sample_noise else torch.zeros_like(mean)) * agent.action_mask
            raw_action = (mean + noise) * agent.action_mask
            samples.append({
                "actions": raw_action.sigmoid() * agent.action_mask,
                "raw_actions": raw_action,
                "means": mean,
                "noise": noise,
                "latent": latent[:, i : i + 1] * agent.action_mask,
                "log_probs": agent.log_prob(raw_action, mean),
            })
            features.append(feature)
        return {key: torch.cat([sample[key] for sample in samples], dim=1) for key in samples[0]}, torch.cat(features, dim=1)

    def evaluate_agent(self, index, inputs, raw_actions, indices, bptt_steps=None):
        """Evaluate agent."""
        agent = self.agents[index]
        features = agent.encode_sequence(inputs[:, :, index : index + 1], bptt_steps).reshape(-1, 1, self.hidden_dim)
        mean = agent.mean(features[indices])
        return {"log_probs": agent.log_prob(raw_actions[indices, index : index + 1], mean)[:, 0], "entropy": agent.entropy()[0]}

    def evaluate_actions(self, inputs, raw_actions, indices, bptt_steps=None):
        parts = [self.evaluate_agent(i, inputs, raw_actions, indices, bptt_steps) for i in range(self.n_agents)]
        return {
            "log_probs": torch.stack([part["log_probs"] for part in parts], dim=1),
            "entropy": torch.stack([part["entropy"] for part in parts]),
        }
