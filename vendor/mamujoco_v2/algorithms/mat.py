"""Mat."""

import torch
from torch import nn

from models.common_critic import SharedCritic
from models.mat import MATActor
from utils.buffer import normalize_advantages
from utils.normalization import Normalizers


class MAT:
    def __init__(self, env_info, config):
        self.config = config
        self.device = torch.device(config["device"])
        input_dim = env_info["obs_dim"] + (env_info["n_agents"] if config["network"]["use_agent_id"] else 0)
        self.actor = MATActor(input_dim, env_info, config).to(self.device)
        self.normalizers = Normalizers(env_info, config, self.device)
        self.critic = SharedCritic(env_info, config, self.normalizers, self.device)
        training = config["training"]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=training["actor_lr"], eps=training["optimizer_eps"])
        self.total_steps = 0
        self.updates = 0

    def init_hidden(self, batch_size):
        return self.actor.init_hidden(batch_size)

    def actor_inputs(self, observations):
        return self.normalizers.actor_inputs(observations, self.config["network"]["use_agent_id"])

    @torch.no_grad()
    def act(self, inputs, hidden, latent, sample_noise):
        return self.actor.sample_actions(inputs, hidden, latent, sample_noise)

    def train(self, mode=True):
        self.actor.train(mode)
        self.critic.train(mode)

    def update(self, batch):
        training = self.config["training"]
        self.train(True)
        stats = {}
        record = lambda name, value: stats.setdefault(name, []).append(float(value))
        valid = batch.filled.squeeze(-1).bool()
        valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
        rewards = self.normalizers.normalize_rewards(batch)
        raw = batch.raw_actions.flatten(0, 1)
        old_log_probs = batch.log_probs.flatten(0, 1)
        evaluate = lambda indices: self.actor.evaluate_actions(batch.inputs, raw, indices, training["bptt_steps"])
        with torch.no_grad():
            record("initial_ratio_error", (evaluate(valid_indices)["log_probs"] - old_log_probs[valid_indices]).expm1().abs().max())

        fixed_targets = self.critic.gae(batch, rewards, use_target=False) if training["gae_value"] == "online" else None
        for _ in range(training["epochs"]):
            advantages, returns = fixed_targets or self.critic.gae(batch, rewards, use_target=True)
            critic_loss, critic_grad = self.critic.update(batch, returns, valid)
            record("critic_loss", critic_loss)
            record("critic_grad_norm", critic_grad)
            if training["normalize_advantages"]:
                advantages = normalize_advantages(advantages, valid)
            flat_advantages = advantages.flatten(0, 1)
            permutation = valid_indices[torch.randperm(valid_indices.numel(), device=valid_indices.device)]
            for minibatch in permutation.split(training["minibatch_size"]):
                current = evaluate(minibatch)
                log_ratio = current["log_probs"] - old_log_probs[minibatch]
                ratio = log_ratio.clamp(-training["log_ratio_clip"], training["log_ratio_clip"]).exp()
                clipped = ratio.clamp(1 - training["clip_ratio"], 1 + training["clip_ratio"])
                advantage = flat_advantages[minibatch]
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                actor_loss = policy_loss - training["entropy_coef"] * current["entropy"].sum()
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Non-finite actor loss; check sigma and learning rate")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), training["grad_norm_clip"], error_if_nonfinite=True)
                self.actor_optimizer.step()
                with torch.no_grad():
                    record("actor_loss", actor_loss)
                    record("policy_loss", policy_loss)
                    record("actor_grad_norm", actor_grad)
                    record("clip_fraction", (((advantage > 0) & (ratio > clipped)) | ((advantage < 0) & (ratio < clipped))).float().mean())
                    record("ratio_mean", ratio.mean())
                    record("approx_kl", (ratio - 1 - log_ratio).mean())

        self.normalizers.update_inputs(batch)
        self.critic.soft_update()
        with torch.no_grad():
            mask = self.actor.action_mask
            record("sigma_mean", (self.actor.sigma() * mask).sum() / mask.sum())
        self.total_steps += batch.env_steps
        self.updates += 1
        return {name: sum(values) / len(values) for name, values in stats.items()}

    def checkpoint_state(self):
        return {"actor": self.actor.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                **self.critic.state_dict(), "normalizers": self.normalizers.state_dict(),
                "total_steps": self.total_steps, "updates": self.updates}

    def load_checkpoint_state(self, state, resume):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state, resume)
        self.normalizers.load_state_dict(state["normalizers"])
        self.total_steps, self.updates = int(state["total_steps"]), int(state["updates"])
        if resume:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
