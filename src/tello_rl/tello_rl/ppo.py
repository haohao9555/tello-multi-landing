"""Ppo."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tello_rl.common import EpisodeLogger, make_env, mlp, orthogonal_init, parse_args, set_seed

# ==================================
# PPO Hyperparameters
# ==================================
PPO_LR = 3e-4
GAMMA = 0.99
PPO_ROLLOUT_STEPS = 2048
PPO_MINIBATCH_SIZE = 64
PPO_UPDATE_EPOCHS = 10

PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_ENT_COEF = 0.0
PPO_MAX_GRAD_NORM = 0.5
LOG_STD_INIT = -0.5

EVAL_INTERVAL = 50_000


# actor network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))
        orthogonal_init(self.mu, 0.01)

    def forward(self, x):
        mu = self.mu(x.float())
        return Normal(mu, self.log_std.exp().expand_as(mu))


# critic network
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v = mlp(obs_dim, 1)
        orthogonal_init(self.v, 1.0)

    def forward(self, x):
        return self.v(x.float())


# rolloutbuffer
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.episode_ends = []

    def add(self, obs, action, reward, log_prob, value, next_value, terminated, truncated):
        self.obs.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.episode_ends.append(bool(terminated or truncated))

    def __len__(self):
        return len(self.rewards)


# gae
def compute_gae(buffer, gamma=GAMMA, gae_lambda=PPO_GAE_LAMBDA):
    rewards = np.array(buffer.rewards, dtype=np.float32)
    values = np.array(buffer.values, dtype=np.float32)
    next_values = np.array(buffer.next_values, dtype=np.float32)
    terminated = np.array(buffer.terminated, dtype=np.float32)
    episode_ends = np.array(buffer.episode_ends, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        gae = delta + gamma * gae_lambda * (1 - episode_ends[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# update
def ppo_update(buffer, actor, critic, actor_optimizer, critic_optimizer, device,
               clip_eps=PPO_CLIP_EPS, update_epochs=PPO_UPDATE_EPOCHS,
               minibatch_size=PPO_MINIBATCH_SIZE):
    advantages, returns = compute_gae(buffer)
    obs = torch.as_tensor(np.stack(buffer.obs), device=device)
    actions = torch.as_tensor(np.stack(buffer.actions), device=device)
    old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(returns, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = len(buffer)

    actor_losses, critic_losses, entropies = [], [], []

    for epoch in range(update_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start + minibatch_size]
            dist = actor(obs[idx])
            new_log_prob = dist.log_prob(actions[idx]).sum(-1)

            ratio = torch.exp(new_log_prob - old_log_probs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(-1).mean()
            actor_loss = actor_loss - PPO_ENT_COEF * entropy

            new_values = critic(obs[idx]).squeeze(-1)
            critic_loss = F.mse_loss(new_values, returns[idx])

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), PPO_MAX_GRAD_NORM)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), PPO_MAX_GRAD_NORM)
            critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())

    return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)


def make_policy(checkpoint_path, obs_dim, act_dim, device='cpu'):
    """Make policy."""
    actor = Actor(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device)['actor'])
    actor.eval()

    def policy(obs):
        with torch.no_grad():
            mu = actor(torch.as_tensor(obs, device=device).unsqueeze(0)).mean
        return np.clip(mu.squeeze(0).cpu().numpy(), -1.0, 1.0)
    return policy


def main():
    args = parse_args('ppo', default_steps=300_000)
    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_env(args.env, wind_level=args.wind, constant_wind=args.constant_wind)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low, act_high = env.action_space.low, env.action_space.high

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=PPO_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=PPO_LR)

    buffer = RolloutBuffer()
    logger = EpisodeLogger()
    train_log = []   # (step, actor_loss, critic_loss, entropy)

    obs, _ = env.reset(seed=args.seed)
    num_updates = 0
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        # 1. Actor sample + Critic value
        obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = actor(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            value = critic(obs_tensor)
        action = action.squeeze(0).cpu().numpy()


        next_obs, reward, terminated, truncated, info = env.step(np.clip(action, act_low, act_high))


        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                next_value = critic(torch.as_tensor(next_obs, device=device).unsqueeze(0)).item()

        # 4. Store transition
        buffer.add(obs, action, reward, log_prob.item(), value.item(), next_value,
                   terminated, truncated)
        logger.step(reward)

        # 5. Reset if episode ends
        if terminated or truncated:
            logger.end_episode(step, info)
            obs, _ = env.reset()
        else:
            obs = next_obs

        # 6. PPO Update
        if len(buffer) >= PPO_ROLLOUT_STEPS or step == args.total_steps:
            actor_loss, critic_loss, entropy = ppo_update(
                buffer, actor, critic, actor_optimizer, critic_optimizer, device)
            train_log.append((step, actor_loss, critic_loss, entropy))
            buffer.clear()
            num_updates += 1

        # 7. Logging
        if step % 10_000 == 0:
            print(f'step {step:>8} | updates {num_updates:>4} | {logger.summary()} | '
                  f'σ {actor.log_std.exp().mean().item():.3f} | time {time.time() - start_time:.0f}s',
                  flush=True)

        # 8. Save checkpoint
        if step % EVAL_INTERVAL == 0:
            torch.save({'step': step, 'actor': actor.state_dict(), 'critic': critic.state_dict()},
                       args.save_dir / f'ppo_step{step}.pt')

    # Save Final Model
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()},
               args.save_dir / 'ppo_final.pt')
    training_time = time.time() - start_time
    np.savez(args.save_dir / 'ppo_logs.npz', episode_log=np.array(logger.episode_log),
             train_log=np.array(train_log), training_time=training_time)
    env.close()
    print(f'PPO Training Completed: {training_time / 60:.1f} min -> {args.save_dir}')


if __name__ == '__main__':
    main()
