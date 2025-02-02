from copy import deepcopy
from ai.base_agent import BaseAgent
from game_env.four_in_row import VirtualBaseEnv
import numpy as np


class GreedyAgent(BaseAgent):
    def __init__(self, v_env: VirtualBaseEnv):
        super().__init__()
        self.v_env = v_env

    def get_action(self, s, training=False):
        max_reward = -np.inf
        best_action = None
        for a in self.v_env.actions:
            s2, _, done = self.v_env.step(a)
            r = self.v_env.get_virtual_reward(s1, a, s2)
            if r > max_reward:
                max_reward = r
                best_action = a
            self.v_env.roll_back(a)
        return best_action
