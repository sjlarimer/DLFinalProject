from pettingzoo.butterfly import knights_archers_zombies_v10
from env.wrappers import shape_rewards


# https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/
class KAZEnv:
    def __init__(self):
        self.env = knights_archers_zombies_v10.parallel_env()

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        rewards = shape_rewards(rewards, infos)

        return obs, rewards, terminations, truncations, infos

    def action_space(self):
        return self.env.action_space(self.env.possible_agents[0])