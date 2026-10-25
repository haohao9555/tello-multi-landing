"""Running statistics for observations, global state, rewards and value targets."""

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    def __init__(self, size, device):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, device=device))
        self.register_buffer("var", torch.ones(size, device=device))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64, device=device))

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        batch_count = values.shape[0]
        batch_var = values.var(dim=0, unbiased=batch_count > 1)
        delta = values.mean(dim=0) - self.mean
        previous_count = self.count.item()
        total_count = previous_count + batch_count
        variance_sum = self.var * previous_count + batch_var * batch_count
        variance_sum += delta.square() * previous_count * batch_count / total_count
        self.mean.add_(delta * batch_count / total_count)
        self.var.copy_(variance_sum / total_count)
        self.count.fill_(total_count)

    def normalize(self, values):
        return (values - self.mean) / self.var.sqrt().clamp_min(1e-8)

    def denormalize(self, values):
        return values * self.var.sqrt().clamp_min(1e-8) + self.mean


class Normalizers(nn.Module):
    """Normalizers."""

    def __init__(self, env_info, config, device):
        super().__init__()
        self.settings = config["normalization"]
        self.obs = RunningMeanStd(env_info["obs_dim"], device)
        self.state = RunningMeanStd(env_info["state_dim"], device)
        self.rewards = RunningMeanStd(1, device)
        self.returns = RunningMeanStd(env_info["n_agents"], device)

    def normalize_observations(self, values):
        return self._normalize_input(values, self.obs)

    def normalize_states(self, values):
        return self._normalize_input(values, self.state)

    def _normalize_input(self, values, statistics):
        if not self.settings["observations"]:
            return values
        bound = self.settings["observation_clip"]
        return ((values - statistics.mean) / (statistics.var + 1e-8).sqrt()).clamp(-bound, bound)

    def actor_inputs(self, observations, use_agent_id):
        """Actor inputs."""
        inputs = [self.normalize_observations(observations)]
        if use_agent_id:
            n_agents = observations.shape[-2]
            identities = torch.eye(n_agents, device=observations.device, dtype=observations.dtype)
            inputs.append(identities.expand(*observations.shape[:-2], n_agents, n_agents))
        return torch.cat(inputs, dim=-1)

    def normalize_rewards(self, batch):
        """Normalize rewards."""
        rewards = batch.rewards
        if self.settings["rewards"]:
            self.rewards.update(rewards[batch.filled.squeeze(-1).bool()])
            rewards = self.rewards.normalize(rewards)
        return rewards * batch.filled

    @torch.no_grad()
    def update_inputs(self, batch):
        if not self.settings["observations"]:
            return
        valid = batch.filled.squeeze(-1).bool()
        self.obs.update(batch.observations[:, :-1][valid])
        self.state.update(batch.states[:, :-1][valid])
