"""FPO：Flow Policy Optimization（Berkeley，"Flow Matching Policy Gradients"）。

在 PPO 基础上只换两处：
  1. 策略是 flow：速度场 v_θ(a_τ, τ; s)，从 ε ~ N(0, σ0²) 出发用 Euler 积分 K 步得到动作
       a_{k+1} = a_k + v_θ(a_k, k/K; s) / K
  2. 似然比用条件流匹配（CFM）损失之差代替（CFM 损失 ≈ 负 ELBO）：
       ℓ_θ(τ, ε) = ‖v_θ(a_τ, τ; s) − (a − ε)‖²,   a_τ = τ·a + (1 − τ)·ε
       ratio = exp( L_old − L_new ),  L = 对 N 组 (τ, ε) 取平均
     其余（clip、GAE、critic）和 PPO 完全相同。flow 的熵算不出来，所以没有熵奖励。

Critic / RolloutBuffer / compute_gae 直接复用 ppo.py。
用法：python -m tello_rl.fpo --env fast --total-steps 300000 --seed 1
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tello_rl.common import EpisodeLogger, make_env, mlp, orthogonal_init, parse_args, set_seed
from tello_rl.ppo import Critic, RolloutBuffer, compute_gae

# ==================================
# FPO Hyperparameters（PPO 部分与 ppo.py 相同）
# ==================================
FPO_LR = 3e-4
FPO_ROLLOUT_STEPS = 2048
FPO_MINIBATCH_SIZE = 64
FPO_UPDATE_EPOCHS = 10
FPO_CLIP_EPS = 0.2
FPO_MAX_GRAD_NORM = 0.5

ODE_STEPS = 10               # 采样动作时的 Euler 积分步数 K
N_MC = 8                     # 每个动作采样几组 (τ, ε) 来估计 CFM 损失
SOURCE_STD = 0.6             # 起点噪声 ε 的标准差；初始速度场 ≈ 0 时策略 ≈ N(0, 0.6²)，与 PPO 初始 σ 相同
TIME_EMBED_DIM = 16
LOG_RATIO_CLIP = 5.0         # 数值保护：log ratio 截断到 ±5

EVAL_INTERVAL = 50_000


# flow actor network：速度场 v_θ(a_τ, τ; s)
class FlowActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.net = mlp(obs_dim + act_dim + TIME_EMBED_DIM, act_dim)
        orthogonal_init(self.net, 0.01)   # 初始速度场 ≈ 0：动作 ≈ 起点噪声
        freqs = torch.arange(1, TIME_EMBED_DIM // 2 + 1, dtype=torch.float32) * np.pi
        self.register_buffer('freqs', freqs)

    def velocity(self, obs, a_tau, tau):
        angles = tau[..., None] * self.freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.net(torch.cat([obs.float(), a_tau, emb], dim=-1))

    def sample(self, obs, eps):
        """从起点噪声 eps 积分 K 步得到动作；eps = 0 时是确定性动作（评估用）。"""
        a = eps
        for k in range(ODE_STEPS):
            tau = torch.full(a.shape[:-1], k / ODE_STEPS, device=a.device)
            a = a + self.velocity(obs, a, tau) / ODE_STEPS
        return a

    def cfm_loss(self, obs, action, tau, eps):
        """obs (B, O)、action (B, A)、tau (B, N)、eps (B, N, A)  ->  每组 (τ, ε) 的损失 (B, N)。"""
        n = tau.shape[1]
        obs_n = obs[:, None].expand(-1, n, -1)
        a_n = action[:, None].expand(-1, n, -1)
        a_tau = tau[..., None] * a_n + (1 - tau[..., None]) * eps
        pred = self.velocity(obs_n, a_tau, tau)
        return ((pred - (a_n - eps)) ** 2).sum(-1)


# update
def fpo_update(buffer, actor, critic, actor_optimizer, critic_optimizer, device,
               clip_eps=FPO_CLIP_EPS, update_epochs=FPO_UPDATE_EPOCHS,
               minibatch_size=FPO_MINIBATCH_SIZE):
    advantages, returns = compute_gae(buffer)
    obs = torch.as_tensor(np.stack(buffer.obs), device=device)
    actions = torch.as_tensor(np.stack(buffer.actions), device=device)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(returns, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = len(buffer)

    # 每个动作固定 N 组 (τ, ε)，新旧策略用同一组，保证 L_old − L_new 只反映参数变化
    tau = torch.rand(batch_size, N_MC, device=device)
    eps = torch.randn(batch_size, N_MC, actor.act_dim, device=device) * SOURCE_STD
    with torch.no_grad():
        old_loss = actor.cfm_loss(obs, actions, tau, eps).mean(-1)   # 更新前的参数就是 θ_old

    actor_losses, critic_losses, log_ratios = [], [], []

    for epoch in range(update_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start + minibatch_size]
            new_loss = actor.cfm_loss(obs[idx], actions[idx], tau[idx], eps[idx]).mean(-1)

            log_ratio = (old_loss[idx] - new_loss).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()

            new_values = critic(obs[idx]).squeeze(-1)
            critic_loss = F.mse_loss(new_values, returns[idx])

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), FPO_MAX_GRAD_NORM)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), FPO_MAX_GRAD_NORM)
            critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            log_ratios.append(log_ratio.abs().mean().item())

    return np.mean(actor_losses), np.mean(critic_losses), np.mean(log_ratios)


def make_policy(checkpoint_path, obs_dim, act_dim, device='cpu'):
    """评估用的确定性策略：从 ε = 0 积分。"""
    actor = FlowActor(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device)['actor'])
    actor.eval()

    def policy(obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            a = actor.sample(x, torch.zeros(1, act_dim, device=device))
        return np.clip(a.squeeze(0).cpu().numpy(), -1.0, 1.0)
    return policy


def main():
    args = parse_args('fpo', default_steps=300_000)
    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_env(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low, act_high = env.action_space.low, env.action_space.high

    actor = FlowActor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=FPO_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=FPO_LR)

    buffer = RolloutBuffer()
    logger = EpisodeLogger()
    train_log = []   # (step, actor_loss, critic_loss, |log ratio|)

    obs, _ = env.reset(seed=args.seed)
    num_updates = 0
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        # 1. Flow 采样动作 + Critic value
        obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
        with torch.no_grad():
            eps = torch.randn(1, act_dim, device=device) * SOURCE_STD
            action = actor.sample(obs_tensor, eps)
            value = critic(obs_tensor)
        action = action.squeeze(0).cpu().numpy()

        # 2. Environment step（裁剪后的动作送进环境，buffer 里存裁剪前的值）
        next_obs, reward, terminated, truncated, info = env.step(np.clip(action, act_low, act_high))

        # 3. Next state value
        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                next_value = critic(torch.as_tensor(next_obs, device=device).unsqueeze(0)).item()

        # 4. Store transition（FPO 不需要 log_prob，填 0）
        buffer.add(obs, action, reward, 0.0, value.item(), next_value, terminated, truncated)
        logger.step(reward)

        # 5. Reset if episode ends
        if terminated or truncated:
            logger.end_episode(step, info)
            obs, _ = env.reset()
        else:
            obs = next_obs

        # 6. FPO Update
        if len(buffer) >= FPO_ROLLOUT_STEPS or step == args.total_steps:
            actor_loss, critic_loss, log_ratio = fpo_update(
                buffer, actor, critic, actor_optimizer, critic_optimizer, device)
            train_log.append((step, actor_loss, critic_loss, log_ratio))
            buffer.clear()
            num_updates += 1

        # 7. Logging
        if step % 10_000 == 0:
            last = train_log[-1][3] if train_log else 0.0
            print(f'step {step:>8} | updates {num_updates:>4} | {logger.summary()} | '
                  f'|log ratio| {last:.3f} | time {time.time() - start_time:.0f}s', flush=True)

        # 8. Save checkpoint
        if step % EVAL_INTERVAL == 0:
            torch.save({'step': step, 'actor': actor.state_dict(), 'critic': critic.state_dict()},
                       args.save_dir / f'fpo_step{step}.pt')

    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()},
               args.save_dir / 'fpo_final.pt')
    training_time = time.time() - start_time
    np.savez(args.save_dir / 'fpo_logs.npz', episode_log=np.array(logger.episode_log),
             train_log=np.array(train_log), training_time=training_time)
    env.close()
    print(f'FPO Training Completed: {training_time / 60:.1f} min -> {args.save_dir}')


if __name__ == '__main__':
    main()
