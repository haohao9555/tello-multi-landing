"""HAPPO（Kuba et al. ICLR 2022；实现对照 PKU-MARL/HARL）：不共享参数的 actor + 公共 centralized critic + 逐 agent 顺序更新。

critic 每个 epoch 回归一步（与 MAPPO 相同）。actor：每次 update 随机打乱 agent 顺序 i_1..i_N，逐样本累积因子 M = 1；
依次对 agent i_m 用 PPO-clip 更新它自己的网络（各自一个 Adam），损失为
    -mean( M · min(r · A, clip(r, 1±ε) · A) )，  r = π_new(a_i|o_i) / π_old(a_i|o_i)
更新完后 M ← M · π_new(a_i|o_i) / π_old(a_i|o_i)（不回传梯度），再更新下一个 agent。
"""

import torch
from torch import nn

from models.common_critic import SharedCritic
from models.happo import HAPPOActor
from utils.buffer import normalize_advantages
from utils.normalization import Normalizers


class HAPPO:
    def __init__(self, env_info, config):
        self.config = config
        self.device = torch.device(config["device"])
        input_dim = env_info["obs_dim"] + (env_info["n_agents"] if config["network"]["use_agent_id"] else 0)
        self.actor = HAPPOActor(input_dim, env_info, config).to(self.device)
        self.normalizers = Normalizers(env_info, config, self.device)
        self.critic = SharedCritic(env_info, config, self.normalizers, self.device)
        training = config["training"]
        self.actor_optimizers = [
            torch.optim.Adam(agent.parameters(), lr=training["actor_lr"], eps=training["optimizer_eps"])
            for agent in self.actor.agents
        ]
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
        evaluate = lambda agent, indices: self.actor.evaluate_agent(agent, batch.inputs, raw, indices, training["bptt_steps"])
        with torch.no_grad():   # 采样时与重算的 log p 应一致，误差应为 0
            current = self.actor.evaluate_actions(batch.inputs, raw, valid_indices, training["bptt_steps"])
            record("initial_ratio_error", (current["log_probs"] - old_log_probs[valid_indices]).expm1().abs().max())

        # critic：与 MAPPO 相同，每个 epoch 一步；优势取第一个 epoch 的估计
        fixed_targets = self.critic.gae(batch, rewards, use_target=False) if training["gae_value"] == "online" else None
        advantages = None
        for _ in range(training["epochs"]):
            epoch_advantages, returns = fixed_targets or self.critic.gae(batch, rewards, use_target=True)
            advantages = epoch_advantages if advantages is None else advantages
            critic_loss, critic_grad = self.critic.update(batch, returns, valid)
            record("critic_loss", critic_loss)
            record("critic_grad_norm", critic_grad)
        if training["normalize_advantages"]:
            advantages = normalize_advantages(advantages, valid)
        flat_advantages = advantages.flatten(0, 1)

        # actor：随机顺序逐 agent 更新，乘上前序 agent 的累积 ratio
        n_agents = self.actor.n_agents
        factor = torch.ones(old_log_probs.shape[0], device=self.device)
        order = range(n_agents) if self.config["fixed_order"] else torch.randperm(n_agents).tolist()
        for agent in order:
            old = old_log_probs[:, agent]
            parameters = list(self.actor.agents[agent].parameters())
            for _ in range(training["epochs"]):
                permutation = valid_indices[torch.randperm(valid_indices.numel(), device=valid_indices.device)]
                for minibatch in permutation.split(training["minibatch_size"]):
                    current = evaluate(agent, minibatch)
                    log_ratio = current["log_probs"] - old[minibatch]
                    ratio = log_ratio.clamp(-training["log_ratio_clip"], training["log_ratio_clip"]).exp()
                    clipped = ratio.clamp(1 - training["clip_ratio"], 1 + training["clip_ratio"])
                    advantage = flat_advantages[minibatch, agent]
                    policy_loss = -(factor[minibatch] * torch.minimum(ratio * advantage, clipped * advantage)).mean()
                    actor_loss = policy_loss - training["entropy_coef"] * current["entropy"]
                    if not torch.isfinite(actor_loss):
                        raise FloatingPointError("actor loss 非有限；请检查 sigma 和学习率")
                    self.actor_optimizers[agent].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad = nn.utils.clip_grad_norm_(parameters, training["grad_norm_clip"], error_if_nonfinite=True)
                    self.actor_optimizers[agent].step()
                    with torch.no_grad():
                        record("actor_loss", actor_loss)
                        record("policy_loss", policy_loss)
                        record("actor_grad_norm", actor_grad)
                        record("clip_fraction", (((advantage > 0) & (ratio > clipped)) | ((advantage < 0) & (ratio < clipped))).float().mean())
                        record("ratio_mean", ratio.mean())
                        record("approx_kl", (ratio - 1 - log_ratio).mean())
            with torch.no_grad():
                new = evaluate(agent, valid_indices)["log_probs"]
                factor[valid_indices] = factor[valid_indices] * (new - old[valid_indices]).exp()
        record("happo_factor_mean", factor[valid_indices].mean())

        self.normalizers.update_inputs(batch)
        self.critic.soft_update()
        with torch.no_grad():
            mask = self.actor.action_mask
            record("sigma_mean", (self.actor.sigma() * mask).sum() / mask.sum())
        self.total_steps += batch.env_steps
        self.updates += 1
        return {name: sum(values) / len(values) for name, values in stats.items()}

    def checkpoint_state(self):
        return {"actor": self.actor.state_dict(),
                "actor_optimizer": [optimizer.state_dict() for optimizer in self.actor_optimizers],
                **self.critic.state_dict(), "normalizers": self.normalizers.state_dict(),
                "total_steps": self.total_steps, "updates": self.updates}

    def load_checkpoint_state(self, state, resume):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state, resume)
        self.normalizers.load_state_dict(state["normalizers"])
        self.total_steps, self.updates = int(state["total_steps"]), int(state["updates"])
        if resume:
            for optimizer, saved in zip(self.actor_optimizers, state["actor_optimizer"]):
                optimizer.load_state_dict(saved)
