

class BaseEnv:
    def __init__(self):
        self.init_state = 0
        self.state = 0
        self.done = False

    def step(self, action):
        s1 = self.state
        s2 = self.get_next_state(action)
        reward = self.get_reward(s1, action, s2)
        done = self.is_game_over()
        return s1, reward, s2, done

    def get_next_state(self, action):
        return self.state

    def get_reward(self, s1, a, s2):
        return 0

    def is_game_over(self):
        return False

    def restart(self):
        self.state = self.init_state
