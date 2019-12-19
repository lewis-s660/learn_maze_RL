import sys

import numpy as np
import tensorflow as tf

from agent_base import AgentBase


class AgentDynamicPrograming(AgentBase):
    def __init__(self, environment, epsilon=0.1, decay=0.9, mode_table=True, size=(8, 8)):
        """
        コンストラクタ
        :param epsilon: ε-Greedy方策で使用するεの値
        :param decay: 行動価値Qを算出する際の減衰率
        :param mode_table: テーブルモード選択フラグ(True:テーブルモード,False:ニューラルネットワークモード)
        :param size: 状態サイズ
        """
        super().__init__()
        self.__environment = environment
        self.__epsilon = epsilon
        self.__decay = decay
        self.__mode_table = mode_table
        self.__size = np.array(size)

        if self.__mode_table:
            # テーブルモードの場合
            try:
                # テーブルの情報をファイルから読み込み
                self.__q_data = np.load('data\\dynamic_programing\\q_data.npy')
                self.__q_count = np.load('data\\dynamic_programing\\q_count.npy')
            except:
                # ファイルからの読み込みに失敗した場合はすべて0で領域を確保
                self.__q_data = np.zeros([self.__size[0], self.__size[1], 4])
                self.__q_count = np.ones(self.__q_data.shape)
        else:
            # ニューラルネットワークモードの場合
            try:
                # モデルをファイルから読み込み
                self.__model = tf.keras.models.load_model('data\\dynamic_programing\\model.hdf5')
            except:
                # モデルの読み込みに失敗した場合はモデルを生成
                self.__model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, input_shape=(3, ), activation='relu'),
                                                           #tf.keras.layers.Dense(1024, activation='relu'),
                                                           tf.keras.layers.Dense(16, activation='relu'),
                                                           tf.keras.layers.Dense(1)])
                self.__model.compile(optimizer='adam', loss='mse')
                # 生成したモデルを保存
                tf.keras.models.save_model(self.__model, 'data\\dynamic_programing\\model.hdf5')
            # 使用するモデルの概要を出力
            self.__model.summary()
            try:
                # 重みをファイルから読み込み
                self.__model.load_weights('data\\dynamic_programing\\weights.hdf5')
            except:
                # 重みの読み込みに失敗した場合は何もしない
                pass

    def get_action(self, status):
        """
        行動取得処理
        状態から行動を決定して返す
        :param status: 状態
        :return: 行動
        """
        if 0 < self.__count_random_policy:
            # ランダム方策で行動選択をする場合
            action = np.random.randint(0, 4)
        else:
            # ε-Greedy方策で行動を選択する場合
            # εと比較するための値を取得
            value = np.random.rand()

            # 現在の状態における各行動での最大の行動価値Qを取得処理
            q_max = self.__get_q(status, 0)
            action = 0
            for i in range(1, 4):
                q = self.__get_q(status, i)
                if q_max < q:
                    # 今までの中で最大の行動価値Qを記憶
                    q_max = q
                    action = i

            if value < self.__epsilon:
                # ランダムで行動を決定する場合
                # 行動価値Qが最大となる行動以外からランダムに行動を選択
                action = (np.random.randint(action + 1, action + 4)) % 4

        return action

    def get_reward(self, status, action, can_action, status_next, is_play, score, actions_effective_next=None):
        """
        報酬取得処理
        行動とその前後の状態などの情報から報酬を決定して返す
        :param status: 行動前の状態
        :param action: 行動
        :param can_action:　選択した行動の有効性
        :param status_next: 行動後の状態
        :param is_play: プレイ状況
        :param score:　スコア
        :param actions_effective_next: 行動後の状態で選択可能な行動のリスト
        :return: 報酬
        """
        reward = 0

        if (np.array(status) == np.array(status_next)).all():
            # ステータスが変わっていない場合
            reward = -10
        if (len(actions_effective_next) <= 1) or not can_action:
            # 行き止まりまたは壁方向を選択した場合
            reward = -100
        if not is_play:
            # ゴールした場合
            reward = 100000
            # ランダム方策の実施回数をデクリメント
            self.__count_random_policy -= 1

        return reward

    def fit(self, experience, epochs=100, size_batch=20, number=1):
        """
        学習実施処理
        指定のパラメーターで学習を実施する
        :param experience: 経験(学習データ)
        :param epochs: エポック数
        :param size_batch: バッチサイズ
        :param number: 出力用のナンバー(fitの実施回数を想定)
        :return: なし
        """
        # とりうる位置を1次元配列で取得
        indices = np.random.choice(range(self.__size[0] * self.__size[1]), self.__size[0] * self.__size[1], replace=False)

        # 学習を開始
        for i in range(epochs):
            for index in indices:
                # スカラー値を座標に変換
                status = np.array([index // self.__size[0], index % self.__size[0]])
                # 有効な行動リストを取得
                actions = self.__environment.get_actions_effective(status)
                # 行動を取得
                #action = self.get_action(status)
                #while action not in actions:
                #    # 有効リストに含まれている行動が取得できるまでループ
                #    action = self.get_action(status)
                for action in range(4):
                    # 移動先の座標を算出
                    status_next = status.copy()
                    if action == 0:
                        status_next[1] -= 1
                    elif action == 1:
                        status_next[0] += 1
                    elif action == 2:
                        status_next[1] += 1
                    else:
                        status_next[0] -= 1
                    # 報酬を取得
                    reward = self.get_reward(None, None, None, not (status_next == np.array([self.__size[1] - 1, self.__size[0] - 1])).all(), None, actions)

                    # 行動価値Q(実際は価値V)の値を算出(移動先の行動価値Qの平均を価値とした)
                    q = 0
                    for k in range(4):
                        try:
                            q += self.__decay * self.__get_q(status_next, k)
                        except:
                            q += 0
                    q /= 4
                    q += reward

                    if self.__mode_table:
                        # テーブルモードの場合
                        self.__q_data[status[1], status[0], action] = q
                    else:
                        # ニューラルネットワークモードの場合
                        #q = self.__model.predict(np.array([status[0], status[1], action])[np.newaxis, :])[0][0]
                        pass

        if self.__mode_table:
            # テーブルモードの場合
            # デーブルの各値をファイルに保存
            np.save('data\\dynamic_programing\\q_data.npy', self.__q_data)
            np.save('data\\dynamic_programing\\q_count.npy', self.__q_count)
        else:
            # ニューラルネットワークモードの場合
            # 学習を実施
            #self.__model.fit(np.array(train_data), np.array(train_label), epochs=epochs)
            # 学習した重みをファイルに保存
            self.__model.save_weights('data\\dynamic_programing\\weights.hdf5')

    def get_q_table(self):
        """
        行動価値Qのテーブル取得処理
        行動価値Qのテーブルを返す(テーブルがない場合は生成も行う)
        :return: 行動価値Qテーブル(x座標, y座標, 行動)
        """
        # 戻り値用の領域をすべて0で生成
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

    def __get_q(self, status, action):
        """
        行動価値Q取得処理
        行動価値Qを算出して返す
        :param status: 状態
        :param action: 行動
        :return: 行動価値Q
        """
        q = 0
        if self.__mode_table:
            # テーブルモードの場合
            q = self.__q_data[status[1], status[0], action]
        else:
            # ニューラルネットワークモードの場合
            q = self.__model.predict(np.array([status[1], status[0], action])[np.newaxis, :])[0][0]

        return q
