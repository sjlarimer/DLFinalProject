from pettingzoo.butterfly import knights_archers_zombies_v10
from env.wrappers import preprocess_agent, shape_rewards


# https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/
# https://pettingzoo.farama.org/tutorials/sb3/connect_four/
# https://pettingzoo.farama.org/tutorials/sb3/waterworld/
# https://pettingzoo.farama.org/tutorials/sb3/kaz/
# https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/
# https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/


class KAZEnv:
    def __init__(self, **kwargs):
        self.env = knights_archers_zombies_v10.parallel_env(**kwargs)

    def reset(self, seed=None):
        obs, infos = self.env.reset(seed=seed)

        obs = {
            agent: preprocess_agent(agent, ob)
            for agent, ob in obs.items()
        }

        return obs

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        obs = {
            agent: preprocess_agent(agent, ob)
            for agent, ob in obs.items()
        }

        rewards = shape_rewards(rewards, infos)

        return obs, rewards, terminations, truncations, infos

    def action_space(self):
        return self.env.action_space(self.env.possible_agents[0])