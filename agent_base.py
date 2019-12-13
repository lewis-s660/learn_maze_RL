import numpy as np


class AgentBase:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def finalize(self, status, score):
        pass

    def get_action(self, status):
        return np.zeros([2], dtype=np.int)

    def get_reward(self, status, action, status_next, is_play, score):
        return 0

    def adjust_experience(self, experience, score):
        pass

    def fit(self, experience, epochs=100, size_batch=20):
        pass

