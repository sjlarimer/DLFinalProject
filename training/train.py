import torch
from torch.distributions import Categorical

from env.kaz_env import KAZEnv
from models.policy import Policy
from agents.ppo_agent import PPOAgent

# https://pettingzoo.farama.org/tutorials/sb3/kaz/
# https://www.geeksforgeeks.org/deep-learning/pettingzoo-multi-agent-reinforcement-learning/
# https://pettingzoo.farama.org/content/tutorials/

env = KAZEnv()
obs = env.reset()

action_dim = env.action_space().n

policy = Policy(action_dim)
agent = PPOAgent(policy)

for episode in range(500):

    obs = env.reset()

    log_probs = []
    old_log_probs = []
    values = []
    rewards_all = []

    done = False

    while not done:
        actions = {}

        for agent_id, ob in obs.items():
            ob = torch.tensor(ob).permute(2, 0, 1).unsqueeze(0).float()

            logits, value = policy(ob)
            dist = Categorical(logits=logits)

            action = dist.sample()

            actions[agent_id] = action.item()

            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)
            old_log_probs.append(log_prob.detach())
            values.append(value)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for r in rewards.values():
            rewards_all.append(r)

        dones = {agent: terminations[agent] or truncations[agent] for agent in terminations}
        done = all(dones.values())

    agent.update(log_probs, values, rewards_all, old_log_probs)

    print(f"Episode {episode} done")