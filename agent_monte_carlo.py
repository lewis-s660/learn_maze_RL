import sys

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from agent_base import AgentBase


class AgentMonteCarlo(AgentBase):
    def __init__(self, epsilon=0.1, delta=0.9, mode_table=True, size=(9, 9)):
        super().__init__()
        self.__epsilon = epsilon
        self.__delta = delta
        self.__mode_table = mode_table
        if self.__mode_table:
            try:
                self.__q_data = np.load('data\\monte_carlo\\q_data')
                self.__q_count = np.load('data\\monte_carlo\\q_count')
            except:
                self.__q_data = np.zeros([size[0], size[1], 4])
                self.__q_count = np.ones(self.__q_data.shape)

    def get_action(self, status):
        value = np.random.randn()

        q_max = self.__get_q(status, 0)
        action = 0
        for i in range(1, 4):
            q = self.__get_q(status, i)
            if q_max < q:
                # 今までの中で最大の行動価値Qを記憶
                q_max = q
                action = i

        if value < self.__epsilon:
            # ランダム方策の場合
            # 行動価値Qが最大となる行動以外からランダムに行動を選択
            action = (np.random.randint(action + 1, action + 4)) % 4

        return action

    def adjust_experience(self, experience, score):
        experience['reward'][-1] = 100

    def fit(self, experience, epochs=100, size_batch=20, number=1):
        if self.__mode_table:
            experience = experience[0][0]
            experience['q'] = [0] * len(experience['status'])
            q_total = np.zeros(self.__q_data.shape)
            q_delta_counter = np.zeros(self.__q_data.shape)
            q = 0
            count = len(experience['status'])

            for i in range(count):
                status = experience['status'][-1 * i]
                q = experience['reward'][-1 * i]
                if i != 0:
                    q += self.__delta * experience['q'][-1 * (i - 1)]
                experience['q'][-1 * i] = q
                q_total[status[0], status[1], experience['action']] += q
                q_delta_counter[status[0], status[1], experience['action']] += 1

                if i % 100 == 0:
                    if i != 0:
                        sys.stdout.flush()
                    print('学習回数：{0}, 進捗率：{1:.3f}'.format(number, i * 100 / count))

            q_total += self.__q_data * self.__q_count
            self.__q_count += q_delta_counter
            self.__q_data = q_total / self.__q_count

            np.save('data\\monte_carlo\\q_data', self.__q_data)
            np.save('data\\monte_carlo\\q_count', self.__q_count)

    def __get_q(self, status, action):
        q = 0

        if self.__mode_table:
            q = self.__q_data[status[0], status[1], action]

        return q



