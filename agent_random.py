import numpy as np

from agent_base import AgentBase


class AgentRandom(AgentBase):
    def __init__(self):
        super().__init__()

    def get_action(self, status):
        return np.random.randint(0, 4)
