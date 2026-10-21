"""TD3（连续动作，离策略）—— 由 ATARI/CARTPOLE.ipynb 的手写 DQN 改来。

和 DQN 的区别：
  - Qnet(s) 输出每个动作的 Q  ->  两个 Critic Q1(s,a)、Q2(s,a) + 一个 Actor μ(s)（tanh 限制在 [-1, 1]）
  - TD 目标：r + γ·max Q'  ->  r + γ·min(Q1', Q2')(s', μ'(s') + 截断噪声)；只用 terminated，超时不算结束
  - 目标网络：每 TARGET_UPDATE 步整体复制  ->  每次更新做软更新（τ = 0.005）
  - Actor 延迟更新：每 POLICY_DELAY 次 critic 更新才更新一次 actor
  - 探索：ε-greedy  ->  在 actor 输出上加高斯噪声
  - 去掉 np.sign 奖励裁剪和 life_lost

用法：python -m tello_rl.td3 --env fast --total-steps 150000 --seed 1
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tello_rl.common import EpisodeLogger, make_env, mlp, parse_args, set_seed

# ==================================
# TD3 Hyperparameters
# ==================================
LR = 3e-4
GAMMA = 0.99
BUFFER_SIZE = 300_000
LEARNING_STARTS = 5_000      # 之前用随机动作填 buffer
BATCH_SIZE = 256
TAU = 0.005                  # 目标网络软更新系数
POLICY_DELAY = 2
EXPL_NOISE = 0.2             # 探索噪声 σ（0.1 时容易卡在悬停不降落）
TARGET_NOISE = 0.2           # 目标策略平滑噪声 σ
NOISE_CLIP = 0.5

EVAL_INTERVAL = 50_000


# actor network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)

    def forward(self, x):
        return torch.tanh(self.net(x.float()))


# critic network（Q(s, a)：动作作为输入）
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, x, a):
        return self.net(torch.cat([x.float(), a.float()], dim=-1)).squeeze(-1)


# Replaybuffer（CARTPOLE 版，动作改成连续向量，done 改成 terminated）
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda a: torch.as_tensor(a, device=device)
        return (to_t(self.obs[idx]), to_t(self.actions[idx]), to_t(self.rewards[idx]),
                to_t(self.next_obs[idx]), to_t(self.terminated[idx]))


def soft_update(target, source, tau=TAU):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - tau).add_(tau * s)


class TD3:
    def __init__(self, obs_dim, act_dim, device):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=LR)
        self.act_dim = act_dim
        self.device = device
        self.num_updates = 0

    # update
    def update(self, batch):
        obs, a, r, next_obs, terminated = batch
        # calculate TD target
        with torch.no_grad():
            noise = (torch.randn_like(a) * TARGET_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_a = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            next_q = torch.min(self.critic1_target(next_obs, next_a),
                               self.critic2_target(next_obs, next_a))
            target = r + GAMMA * (1 - terminated) * next_q
        # critic loss + update
        q1 = self.critic1(obs, a)
        q2 = self.critic2(obs, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed actor update + soft target update
        self.num_updates += 1
        actor_loss = None
        if self.num_updates % POLICY_DELAY == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.actor_target, self.actor)
            soft_update(self.critic1_target, self.critic1)
            soft_update(self.critic2_target, self.critic2)
            actor_loss = actor_loss.item()
        return critic_loss.item(), actor_loss, q1.mean().item()

    # exploration
    def select_action(self, obs, noise=EXPL_NOISE):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        a = a.squeeze(0).cpu().numpy()
        if noise > 0:
            a = a + np.random.normal(0.0, noise, size=self.act_dim)
        return np.clip(a, -1.0, 1.0)

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict()}


def make_policy(checkpoint_path, obs_dim, act_dim, device='cpu'):
    """评估用的确定性策略：actor 输出，不加噪声。"""
    actor = Actor(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device)['actor'])
    actor.eval()

    def policy(obs):
        with torch.no_grad():
            a = actor(torch.as_tensor(obs, device=device).unsqueeze(0))
        return a.squeeze(0).cpu().numpy()
    return policy


def main():
    args = parse_args('td3', default_steps=150_000)
    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_env(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # 动作统一在 [-1, 1] 里学，送进环境前按动作空间缩放（Tello 环境本来就是 [-1, 1]）
    act_low, act_high = env.action_space.low, env.action_space.high
    to_env = lambda a: act_low + (a + 1.0) * 0.5 * (act_high - act_low)

    agent = TD3(obs_dim, act_dim, device)
    buffer = ReplayBuffer(BUFFER_SIZE, obs_dim, act_dim)
    logger = EpisodeLogger()
    train_log = []   # (step, critic_loss, actor_loss, q_mean)

    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    start = time.time()

    for step in range(1, args.total_steps + 1):
        if step <= LEARNING_STARTS:
            a = np.random.uniform(-1.0, 1.0, size=act_dim)
        else:
            a = agent.select_action(obs)

        next_obs, r, terminated, truncated, info = env.step(to_env(a))
        logger.step(r)
        buffer.add(obs, a, r, next_obs, float(terminated))
        obs = next_obs

        if terminated or truncated:
            logger.end_episode(step, info)
            obs, _ = env.reset()

        if step > LEARNING_STARTS:
            critic_loss, actor_loss, q_mean = agent.update(buffer.sample(BATCH_SIZE, device))
            if step % 1000 == 0:
                train_log.append((step, critic_loss, actor_loss or np.nan, q_mean))

        if step % 10_000 == 0:
            print(f'step {step:>8} | {logger.summary()} | {time.time() - start:6.0f}s', flush=True)

        # Save Checkpoint
        if step % EVAL_INTERVAL == 0:
            torch.save({'step': step, **agent.state_dict()}, args.save_dir / f'td3_step{step}.pt')

    # Save Final Model
    torch.save(agent.state_dict(), args.save_dir / 'td3_final.pt')
    training_time = time.time() - start
    np.savez(args.save_dir / 'td3_logs.npz', episode_log=np.array(logger.episode_log),
             train_log=np.array(train_log), training_time=training_time)
    env.close()
    print(f'TD3 Training Completed: {training_time / 60:.1f} min -> {args.save_dir}')


if __name__ == '__main__':
    main()
