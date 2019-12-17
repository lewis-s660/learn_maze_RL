import sys

import numpy as np
import tensorflow as tf

from agent_base import AgentBase


class AgentDynamicPrograming(AgentBase):
    def __init__(self, environment, epsilon=0.1, decay=0.9, mode_table=True, size=(8, 8)):
        super().__init__()
        self.__environment = environment
        self.__epsilon = epsilon
        self.__decay = decay
        self.__mode_table = mode_table
        self.__size = size
        if self.__mode_table:
            # テーブルモードの場合
            try:
                self.__q_data = np.load('data\\dynamic_programing\\q_data.npy')
                self.__q_count = np.load('data\\dynamic_programing\\q_count.npy')
            except:
                self.__q_data = np.zeros([self.__size[0], self.__size[1], 4])
                self.__q_count = np.ones(self.__q_data.shape)
        else:
            # ニューラルネットワークモードの場合
            self.__model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, input_shape=(3, ), activation='relu'),
                                                       tf.keras.layers.Dense(1024, activation='relu'),
                                                       tf.keras.layers.Dense(128, activation='relu'),
                                                       tf.keras.layers.Dense(1)])
            self.__model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            self.__model.summary()
            try:
                self.__model.load_weights('data\\dynamic_programing\\weights.hdf5')
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
        # とりうる位置を1次元配列で取得
        indices = np.random.choice(range(self.__size[0] * self.__size[1]), replace=False)

        # 学習を開始
        for i in range(epochs):
            for index in indices:
                # スカラー値を座標に変換
                status = np.array([index // self.__size[0], index % self.__size[0]])
                # 有効な行動リストを取得
                actions = self.__environment.get_actions_effective(status)
                # 行動を取得
                action = self.get_action(status)
                while action not in actions:
                    # 有効リストに含まれている行動が取得できるまでループ
                    action = self.get_action(status)
                # 報酬を取得
                reward = self.get_reward(None, None, None, False if status == np.array([self.__size[1], self.__size[0]]) else True, None, actions)
                q = reward + self.__decay * self.__get_q(status, action)
                if self.__mode_table:
                    # テーブルモードの場合
                    self.__q_data[status[0], status[1], action] = q
                else:
                    # ニューラルネットワークモードの場合
                    #q = self.__model.predict(np.array([status[0], status[1], action])[np.newaxis, :])[0][0]
                    pass

        if self.__mode_table:
            # テーブルモードの場合
            np.save('data\\dynamic_programing\\q_data.npy', self.__q_data)
            np.save('data\\dynamic_programing\\q_count.npy', self.__q_count)
        else:
            # ニューラルネットワークモードの場合
            #self.__model.fit(np.array(train_data), np.array(train_label), epochs=epochs)
            self.__model.save_weights('data\\dynamic_programing\\weights.hdf5')

    def __get_q(self, status, action):
        q = 0

        if self.__mode_table:
            # テーブルモードの場合
            q = self.__q_data[status[0], status[1], action]
        else:
            # ニューラルネットワークモードの場合
            q = self.__model.predict(np.array([status[0], status[1], action])[np.newaxis, :])[0][0]

        return q

    def get_q_table(self):
        q_data = np.zeros([self.__size[0], self.__size[1], 4])
        if self.__mode_table:
            # テーブルモードの場合
            q_data = self.__q_data.copy()
        else:
            # ニューラルネットワークモードの場合
            for i in range(q_data.shape[0]):
                for j in range(q_data.shape[1]):
                    for k in range(q_data.shape[2]):
                        q_data[i, j, k] = self.__get_q((i, j), k)

        return q_data
