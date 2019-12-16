import sys

import numpy as np
import tensorflow as tf

from agent_base import AgentBase


class AgentMonteCarlo(AgentBase):
    def __init__(self, epsilon=0.1, decay=0.9, mode_table=True, size=(9, 9)):
        super().__init__()
        self.__epsilon = epsilon
        self.__decay = decay
        self.__mode_table = mode_table
        if self.__mode_table:
            # テーブルモードの場合
            try:
                self.__q_data = np.load('data\\monte_carlo\\q_data.npy')
                self.__q_count = np.load('data\\monte_carlo\\q_count.npy')
            except:
                self.__q_data = np.zeros([size[0], size[1], 4])
                self.__q_count = np.ones(self.__q_data.shape)
        else:
            # ニューラルネットワークモードの場合
            self.__model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, input_shape=(3, ), activation='relu'),
                                                       tf.keras.layers.Dense(128, activation='relu'),
                                                       tf.keras.layers.Dense(1)])
            self.__model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            self.__model.summary()
            try:
                self.__model.load_weights('data\\monte_carlo\\weights.hdf5')
            except:
                pass

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

    def get_reward(self, status, action, status_next, is_play, score, actions_effective_next=None):
        reward = 0
        if len(actions_effective_next) <= 1:
            # 行き止まりの場合
            reward = -10
        if not is_play:
            # ゴールした場合
            reward = 100

        return reward

    def fit(self, experience, epochs=100, size_batch=20, number=1):
        experience = experience[0][0]
        q = 0
        count = len(experience['status'])
        if self.__mode_table:
            # テーブルモードの場合
            q_total = np.zeros(self.__q_data.shape)
            q_delta_counter = np.zeros(self.__q_data.shape)
        else:
            # ニューラルネットワークモードの場合
            train_data = list()
            train_label = list()

        for i in range(count):
            status = experience['status'][-1 * i]
            action = experience['action'][-1 * i]
            q = experience['reward'][-1 * i] + self.__decay * q
            if self.__mode_table:
                # テーブルモードの場合
                q_total[status[0], status[1], action] += q
                q_delta_counter[status[0], status[1], action] += 1
            else:
                # ニューラルネットワークモードの場合
                train_data.append((status[0], status[1], action))
                train_label.append(q)

        if self.__mode_table:
            # テーブルモードの場合
            q_total += self.__q_data * self.__q_count
            self.__q_count += q_delta_counter
            self.__q_data = q_total / self.__q_count

            np.save('data\\monte_carlo\\q_data.npy', self.__q_data)
            np.save('data\\monte_carlo\\q_count.npy', self.__q_count)
        else:
            # ニューラルネットワークモードの場合
            self.__model.fit(np.array(train_data), np.array(train_label), epochs=epochs)
            self.__model.save_weights('data\\monte_carlo\\weights.hdf5')

    def __get_q(self, status, action):
        q = 0

        if self.__mode_table:
            # テーブルモードの場合
            q = self.__q_data[status[0], status[1], action]
        else:
            # ニューラルネットワークモードの場合
            q = self.__model.predict(np.array([status[0], status[1], action])[np.newaxis, :])

        return q



