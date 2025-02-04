from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        self.training = False

    def learn(self, *args):
        # one-step of the agent
        pass

    @abstractmethod
    def get_action(self, s):
        # choose an action under the state s
        pass

    def set_train_mode(self):
        self.training = True

    def set_eval_mode(self):
        self.training = False
