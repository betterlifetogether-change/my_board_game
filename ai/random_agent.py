from ai.base_agent import BaseAgent
import random


class RandomAgent(BaseAgent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def get_action(self, s):
        return random.choice(range(self.num_actions))
