from copy import deepcopy

class BaseEnv:
    def __init__(self):
        self.init_state = 0
        self.state = 0
        self.done = False
        self.actions = set()

    def step(self, action):
        s1 = self.get_cur_state()
        if self.is_game_over():
            return s1, 0, True
        s2 = self.get_next_state(action)
        reward = self.get_reward(s1["state"], action, s2["state"])
        done = self.is_game_over()
        return s2, reward, done

    def get_cur_state(self):
        return {"state": deepcopy(self.state)}

    def get_next_state(self, action):
        return {"state": deepcopy(self.state)}

    def get_reward(self, s1, a, s2):
        return 0

    def is_game_over(self):
        return False

    def restart(self):
        self.state = self.init_state


class VirtualBaseEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def roll_back(self, action):
        return

    def get_virtual_reward(self, s1, a, s2):
        return 0
