import os

import numpy as np
import tensorflow as tf

from agent_base import AgentBase


class AgentTD(AgentBase):
    def __init__(self, environment, mode_sarsa, epsilon=0.1, decay=0.9, eta=0.1, gradient_minimum=0.001, mode_table=True, size=(8, 8), count_random_policy=0):
        """
        コンストラクタ
        :param environment: 環境
        :param mode_sarsa: SARSAフラグ
        :param epsilon: ε-Greedy方策で使用するεの値
        :param decay: 行動価値Qを算出する際の減衰率
        :param eta: 学習率
        :param gradient_minimum: 学習が必要となる最小の勾配
        :param mode_table: テーブルモード選択フラグ(True:テーブルモード,False:ニューラルネットワークモード)
        :param size: 状態サイズ
        :param count_random_policy: ランダム方策実施回数
        """
        super().__init__()
        self.__environment = environment
        self.__mode_sarsa = mode_sarsa
        self.__epsilon = epsilon
        self.__decay = decay
        self.__eta = eta
        self.__gradient_minimum = gradient_minimum
        self.__mode_table = mode_table
        self.__size = np.array(size)
        self.__count_random_policy = count_random_policy
        self.__action = None
        path_directory = 'data\\td'

        if self.__mode_table:
            # テーブルモードの場合
            if self.__mode_sarsa:
                # SARSAモードの場合
                self.__path_data = os.path.join(path_directory, "q_data_sarsa.npy")
            else:
                # SARSAモードでない場合
                self.__path_data = os.path.join(path_directory, "q_data.npy")

            try:
                # テーブルの情報をファイルから読み込み
                self.__q_data = np.load(self.__path_data)
            except:
                # ファイルからの読み込みに失敗した場合はすべて0で領域を確保
                self.__q_data = np.zeros([self.__size[0], self.__size[1], 4])
        else:
            # ニューラルネットワークモードの場合
            if self.__mode_sarsa:
                # SARSAモードの場合
                self.__path_model = os.path.join(path_directory, "model_sarsa.hdf5")
                self.__path_weights = os.path.join(path_directory, "weights_sarsa.hdf5")
            else:
                # SARSAモードでない場合
                self.__path_model = os.path.join(path_directory, "model.hdf5")
                self.__path_weights = os.path.join(path_directory, "weights.hdf5")

            try:
                # モデルをファイルから読み込み
                self.__model = tf.keras.models.load_model(self.__path_model)
            except:
                # モデルの読み込みに失敗した場合はモデルを生成
                self.__model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, input_shape=(3, ), activation='relu'),
                                                           tf.keras.layers.Dense(128, activation='relu'),
                                                           tf.keras.layers.Dense(16, activation='relu'),
                                                           tf.keras.layers.Dense(1)])
                self.__model.compile(optimizer='adam', loss='mse')
                # 生成したモデルを保存
                tf.keras.models.save_model(self.__model, self.__path_model)
            # 使用するモデルの概要を出力
            self.__model.summary()

            try:
                # 重みをファイルから読み込み
                self.__model.load_weights(self.__path_weights)
            except:
                # 重みの読み込みに失敗した場合は何もしない
                pass

    @property
    def mode_sarsa(self):
        """
        SARSAモードフラグ
        :return: SARSAモードフラグ(True:SARSA,False:SARSA以外)
        """
        return self.__mode_sarsa

    def get_action(self, status, actions_effective, is_previous=False):
        """
        行動取得処理
        状態から行動を決定して返す
        :param status: 状態
        :param actions_effective: 有効行動リスト
        :param is_previous: 前回取得値取得フラグ
        :return: 行動
        """

        if is_previous and self.__action is not None:
            # 前回取得値の取得かつ前回取得値が存在する場合
            action = self.__action
        else:
            # 新たに行動を取得する必要がある場合
            if 0 < self.__count_random_policy:
                # ランダム方策で行動選択をする場合
                action = np.random.randint(0, 4)
            else:
                # ε-Greedy方策で行動を選択する場合
                # εと比較するための値を取得
                value = np.random.rand()

                # 現在の状態における各行動での最大の行動価値Qを取得処理
                q_max = self.__get_q(status, 0, None)
                action = 0
                for i in range(1, 4):
                    q = self.__get_q(status, i, None)
                    if q_max < q:
                        # 今までの中で最大の行動価値Qを記憶
                        q_max = q
                        action = i

                if value < self.__epsilon:
                    # ランダムで行動を決定する場合
                    # 行動価値Qが最大となる行動以外からランダムに行動を選択
                    action = (np.random.randint(action + 1, action + 4)) % 4
            self.__action = action

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

        if (len(actions_effective_next) <= 1) or not can_action:
            # 行き止まりまたは壁方向を選択した場合
            reward = -100
        if not is_play:
            # ゴールした場合
            reward = 10000
            # ランダム方策の実施回数をデクリメント
            self.__count_random_policy -= 1

        return reward

    def get_q(self, status, action, status_next, action_next, reward):
        """
        報酬取得処理
        行動とその前後の状態などの情報から報酬を決定して返す
        :param status: 行動前の状態
        :param action: 行動
        :param status_next: 行動後の状態
        :param action_next: 行動後の状態の行動
        :param reward: 報酬
        :return: 行動価値Q
        """
        q = self.__get_q(status, action, None)

        q_next = 0
        if self.__mode_sarsa:
            # SARSAモードの場合
            q_next = self.__get_q(status_next, action_next, None)
        else:
            # SARSAモードでない場合
            # 次の状態での最大の行動価値Qを取得
            for i in range(4):
                q_next_tmp = self.__get_q(status_next, i, None)
                if q_next < q_next_tmp:
                    q_next = q_next_tmp

        q = q + self.__eta * ((reward + self.__decay * q_next) - q)

        if self.__mode_table:
            # テーブルモードの場合
            # 行動価値Qを更新
            self.__q_data[status[1], status[0], action] = q

        return q

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

        if self.__mode_table:
            # テーブルモードの場合
            # デーブルの各値をファイルに保存
            np.save(self.__path_data, self.__q_data)
        elif experience is not None:
            # ニューラルネットワークモードかつ学習データが存在する場合
            # 1回のプレイの経験のみを使用する
            # テーブルモードであれば複数プレイのデータを使用しても問題ないが,ニューラルネットワークモードの場合は
            # 手数と減衰率から行動価値Qの値がプレイによって大きく異なることにより,学習が進まない可能性があるため.
            experience = experience[0][0]
            q = 0
            count = len(experience['status'])
            # 学習データを格納するリストを作成
            train_data = list()
            train_label = list()

            for i in range(count):
                # 新しい経験のデータから処理を実施
                status = experience['status'][-1 * i]
                action = experience['action'][-1 * i]
                q = experience['q'][-1 * i]

                train_data.append((status[1], status[0], action))
                train_label.append(q)

            # 学習を実施
            self.__model.fit(np.array(train_data), np.array(train_label), epochs=epochs)
            # 学習した重みをファイルに保存
            self.__model.save_weights(self.__path_weights)

    def get_q_table(self, get_actions_effective):
        """
        行動価値Qのテーブル取得処理
        行動価値Qのテーブルを返す(テーブルがない場合は生成も行う)
        :param get_actions_effective: 有効行動リスト取得ハンドラ
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
                        q_data[i, j, k] = self.__get_q((i, j), k, None)

        return q_data

    def __get_q(self, status, action, actions_effective):
        """
        行動価値Q取得処理
        行動価値Qを算出して返す
        :param status: 状態
        :param action: 行動
        :param actions_effective: 有効行動リスト
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

    def get_q_table_experience(self, experience):
        experience = experience[0][0]
        q = 0
        count = len(experience['status'])
        q_data = np.zeros([self.__size[0], self.__size[1], 4])
        train_data = list()

        for i in range(count):
            status = experience['status'][-1 * i]
            action = experience['action'][-1 * i]
            q = experience['reward'][-1 * i] + self.__decay * q

            is_register = False
            for j in range(len(train_data)):
                if train_data[j] == (status[1], status[0], action):
                    # 登録済みのステータスと行動の組み合わせの場合
                    is_register = True
                    break

            if not is_register:
                # 未登録のステータスと行動の組み合わせの場合
                train_data.append((status[1], status[0], action))
                q_data[status[1], status[0], action] = q

        return q_data
