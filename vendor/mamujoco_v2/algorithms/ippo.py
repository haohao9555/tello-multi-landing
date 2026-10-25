"""Parameter-sharing independent PPO with an observation-only value function.

Actor, PPO objective, GAE, optimisation and reward/return normalisation reuse
the MAPPO implementation. Only the critic information set is local: the same
agent observation supplied to the actor, without privileged global state.
"""
import copy
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from algorithms.mappo import MAPPO
from models.mappo import MAPPOActor
from models.common_critic import SharedCritic
from utils.normalization import Normalizers


class LocalValueNetwork(nn.Module):
    def __init__(self,obs_dim,width):
        super().__init__()
        self.fc1=nn.Linear(obs_dim,width)
        self.fc2=nn.Linear(width,width)
        self.fc3=nn.Linear(width,1)

    def forward(self,observations):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(observations))))).squeeze(-1)


class LocalNormalizers(Normalizers):
    @torch.no_grad()
    def update_inputs(self,batch):
        if self.settings['observations']:
            valid=batch.filled.squeeze(-1).bool()
            self.obs.update(batch.observations[:,:-1][valid])
        # Global-state statistics are deliberately never updated or used.


class CriticObservationNormalizer:
    def __init__(self,normalizers):
        self._normalizers=normalizers
        self.returns=normalizers.returns

    def normalize_states(self,observations):
        return self._normalizers.normalize_observations(observations)


class IndependentCritic(SharedCritic):
    def __init__(self,env_info,config,normalizers,device):
        self.config=config
        self.normalizers=CriticObservationNormalizer(normalizers)
        self.network=LocalValueNetwork(env_info['obs_dim'],config['network']['hidden_dim']).to(device)
        self.target=copy.deepcopy(self.network).requires_grad_(False)
        settings=config['training']
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=settings['critic_lr'],eps=settings['optimizer_eps'])

    @staticmethod
    def observation_batch(batch):
        return SimpleNamespace(states=batch.observations,terminated=batch.terminated,filled=batch.filled)

    def gae(self,batch,rewards,use_target):
        return super().gae(self.observation_batch(batch),rewards,use_target)

    def update(self,batch,returns,valid):
        return super().update(self.observation_batch(batch),returns,valid)


class IPPO(MAPPO):
    def __init__(self,env_info,config):
        self.config=config;self.device=torch.device(config['device'])
        input_dim=env_info['obs_dim']+(env_info['n_agents'] if config['network']['use_agent_id'] else 0)
        self.actor=MAPPOActor(input_dim,env_info,config).to(self.device)
        self.normalizers=LocalNormalizers(env_info,config,self.device)
        self.critic=IndependentCritic(env_info,config,self.normalizers,self.device)
        settings=config['training']
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=settings['actor_lr'],eps=settings['optimizer_eps'])
        self.total_steps=0;self.updates=0
