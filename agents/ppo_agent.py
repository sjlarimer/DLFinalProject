import torch
import torch.nn.functional as F

# https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/

class PPOAgent:
    def __init__(self, policy, lr=1e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.gamma = 0.99
        self.eps_clip = 0.2

    def update(self, log_probs, values, rewards, old_log_probs):
        if isinstance(values, list):
            values = torch.cat(values)
        else:
            values = values

        if isinstance(log_probs, list):
            log_probs = torch.stack(log_probs)

        if isinstance(old_log_probs, list):
            old_log_probs = torch.stack(old_log_probs)

        if isinstance(rewards, list):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        values = values.squeeze()

        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        min_len = min(len(returns), len(values), len(log_probs), len(old_log_probs))

        returns = returns[:min_len]
        values = values[:min_len]
        log_probs = log_probs[:min_len]
        old_log_probs = old_log_probs[:min_len]
        advantages = returns - values.detach()

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = torch.max(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()