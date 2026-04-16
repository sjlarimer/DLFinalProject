import torch
from torch.distributions import Categorical

from env.kaz_env import KAZEnv
from models.policy import Policy
from agents.ppo_agent import PPOAgent

# https://pettingzoo.farama.org/tutorials/sb3/kaz/
# https://www.geeksforgeeks.org/deep-learning/pettingzoo-multi-agent-reinforcement-learning/
# https://pettingzoo.farama.org/content/tutorials/


def train(episodes=500, seed=0, **env_kwargs):
    env = KAZEnv(**env_kwargs)

    action_dim = env.action_space().n
    policy = Policy(action_dim)
    agent = PPOAgent(policy)

    print("Starting training...")

    for episode in range(episodes):
        obs = env.reset(seed=seed)

        log_probs = {a: [] for a in env.env.possible_agents}
        old_log_probs = {a: [] for a in env.env.possible_agents}
        values = {a: [] for a in env.env.possible_agents}
        rewards_all = {a: [] for a in env.env.possible_agents}

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

                log_probs[agent_id].append(log_prob)
                old_log_probs[agent_id].append(log_prob.detach())
                values[agent_id].append(value.squeeze())

            obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                a: terminations[a] or truncations[a]
                for a in terminations
            }

            for agent_id in rewards:
                rewards_all[agent_id].append(rewards[agent_id])

            done = all(dones.values())

        flat_log_probs = []
        flat_old_log_probs = []
        flat_values = []
        flat_rewards = []

        for agent_id in log_probs:
            L = min(
                len(log_probs[agent_id]),
                len(values[agent_id]),
                len(rewards_all[agent_id]),
            )

            flat_log_probs.extend(log_probs[agent_id][:L])
            flat_old_log_probs.extend(old_log_probs[agent_id][:L])
            flat_values.extend(values[agent_id][:L])
            flat_rewards.extend(rewards_all[agent_id][:L])

        flat_values = torch.stack(flat_values)
        flat_log_probs = torch.stack(flat_log_probs)
        flat_old_log_probs = torch.stack(flat_old_log_probs)
        flat_rewards = torch.tensor(flat_rewards, dtype=torch.float32)

        agent.update(flat_log_probs, flat_values, flat_rewards, flat_old_log_probs)

        if episode % 10 == 0:
            print(f"Episode {episode} complete")

    print("Training finished.")
    return policy


def eval(policy, num_games=10, render_mode=None, **env_kwargs):
    env = KAZEnv(**env_kwargs)

    print(f"\nStarting evaluation for {num_games} games...")

    total_rewards = []

    for game in range(num_games):
        obs = env.reset(seed=game)

        done = False
        game_reward = 0

        while not done:
            actions = {}

            for agent_id, ob in obs.items():
                ob = torch.tensor(ob).permute(2, 0, 1).unsqueeze(0).float()

                logits, _ = policy(ob)
                dist = Categorical(logits=logits)

                action = torch.argmax(dist.probs)
                actions[agent_id] = action.item()

            obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                a: terminations[a] or truncations[a]
                for a in terminations
            }

            game_reward += sum(rewards.values())
            done = all(dones.values())

        total_rewards.append(game_reward)

    avg_reward = sum(total_rewards) / len(total_rewards)

    print(f"Average reward over {num_games} games: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=False)

    policy = train(episodes=200, **env_kwargs)

    eval(policy, num_games=10, **env_kwargs)