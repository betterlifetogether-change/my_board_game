from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        return

    def learn(self, *args):
        # one-step of the agent
        pass

    @abstractmethod
    def get_action(self, s, training=False):
        # choose an action under the state s
        pass

