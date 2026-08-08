# agent/ppo_agent.py
# PPO agent for circuit optimization.
# Actor takes 64-dim GNN embedding → action probabilities over 6 mutation rules.
# Critic takes same embedding → scalar value estimate.
# Standard clipped PPO update (Schulman et al. 2017).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ─────────────────────────────────────────────────────────────
# NETWORKS
# ─────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Maps state embedding → action probabilities."""
    def __init__(self, state_dim: int = 64, n_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    """Maps state embedding → scalar value."""
    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# ROLLOUT BUFFER
# ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one episode of transitions for PPO update."""

    def __init__(self):
        self.states   = []
        self.actions  = []
        self.rewards  = []
        self.dones    = []
        self.log_probs = []
        self.values   = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99,
                         gae_lambda: float = 0.95,
                         last_value: float = 0.0):
        """Generalized Advantage Estimation (GAE)."""
        returns    = []
        advantages = []
        gae        = 0.0
        next_val   = last_value

        for reward, done, value in zip(
                reversed(self.rewards),
                reversed(self.dones),
                reversed(self.values)):

            mask   = 0.0 if done else 1.0
            delta  = reward + gamma * next_val * mask - value
            gae    = delta + gamma * gae_lambda * mask * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + value)
            next_val = value

        returns    = torch.tensor(returns,    dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8)

        return returns, advantages


# ─────────────────────────────────────────────────────────────
# PPO AGENT
# ─────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self,
                 state_dim   : int   = 64,
                 n_actions   : int   = 6,
                 lr          : float = 3e-4,
                 gamma       : float = 0.99,
                 gae_lambda  : float = 0.95,
                 clip_eps    : float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef  : float = 0.5,
                 update_epochs: int  = 4):

        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.update_epochs = update_epochs

        self.actor  = Actor(state_dim, n_actions)
        self.critic = Critic(state_dim)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray):
        """
        Given state, samples action and returns
        (action, log_prob, value) for storage.
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = self.actor(state_t)
            value = self.critic(state_t)

        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()

        return (action.item(),
                dist.log_prob(action).item(),
                value.squeeze().item())

    def update(self):
        """PPO clipped surrogate update over stored rollout."""
        states_t    = torch.tensor(
            np.array(self.buffer.states), dtype=torch.float32)
        actions_t   = torch.tensor(
            self.buffer.actions, dtype=torch.long)
        old_lp_t    = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32)

        returns, advantages = self.buffer.compute_returns(
            self.gamma, self.gae_lambda)

        total_loss = 0.0

        for _ in range(self.update_epochs):
            probs    = self.actor(states_t)
            dist     = torch.distributions.Categorical(probs)
            new_lp   = dist.log_prob(actions_t)
            entropy  = dist.entropy().mean()
            values   = self.critic(states_t)

            # PPO clipped objective
            ratio    = torch.exp(new_lp - old_lp_t)
            surr1    = ratio * advantages
            surr2    = torch.clamp(ratio,
                                    1 - self.clip_eps,
                                    1 + self.clip_eps) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)

            loss = (actor_loss
                    + self.value_coef   * critic_loss
                    - self.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) +
                list(self.critic.parameters()), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return round(total_loss / self.update_epochs, 5)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor' : self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
        print(f"[PPO] Saved to {path}") 
    def select_action_greedy(self, state):
        """Deterministic action selection for evaluation — picks highest probability action."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_t)
        action = torch.argmax(probs, dim=-1)
        return action.item()
    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        print(f"[PPO] Loaded from {path}")