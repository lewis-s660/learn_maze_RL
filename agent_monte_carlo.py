import sys

import numpy as np
import tensorflow as tf

from agent_base import AgentBase


class AgentMonteCarlo(AgentBase):
    def __init__(self, epsilon=0.1, decay=0.99, mode_table=True, size=(8, 8), count_random_policy=0):
        """
        コンストラクタ
        :param epsilon: ε-Greedy方策で使用するεの値
        :param decay: 行動価値Qを算出する際の減衰率
        :param mode_table: テーブルモード選択フラグ(True:テーブルモード,False:ニューラルネットワークモード)
        :param size: 状態サイズ
        :param count_random_policy: ランダム方策実施回数
        """
        super().__init__()
        self.__epsilon = epsilon
        self.__decay = decay
        self.__mode_table = mode_table
        self.__size = size
        self.__count_random_policy = count_random_policy

        if self.__mode_table:
            # テーブルモードの場合
            try:
                # テーブルの情報をファイルから読み込み
                self.__q_data = np.load('data\\monte_carlo\\q_data.npy')
                self.__q_count = np.load('data\\monte_carlo\\q_count.npy')
            except:
                # ファイルからの読み込みに失敗した場合はすべて0で領域を確保
                self.__q_data = np.zeros([self.__size[0], self.__size[1], 4])
                self.__q_count = np.ones(self.__q_data.shape)
        else:
            # ニューラルネットワークモードの場合
            try:
                # モデルをファイルから読み込み
                self.__model = tf.keras.models.load_model('data\\monte_carlo\\model.hdf5')
            except:
                # モデルの読み込みに失敗した場合はモデルを生成
                self.__model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, input_shape=(7, ), activation='relu'),
                                                           tf.keras.layers.Dense(128, activation='relu'),
                                                           tf.keras.layers.Dense(16, activation='relu'),
                                                           tf.keras.layers.Dense(1)])
                self.__model.compile(optimizer='adam', loss='mse')
                # 生成したモデルを保存
                tf.keras.models.save_model(self.__model, 'data\\monte_carlo\\model.hdf5')
            # 使用するモデルの概要を出力
            self.__model.summary()
            try:
                # 重みをファイルから読み込み
                self.__model.load_weights('data\\monte_carlo\\weights.hdf5')
            except:
                # 重みの読み込みに失敗した場合は何もしない
                pass

    def get_action(self, status, actions_effective):
        """
        行動取得処理
        状態から行動を決定して返す
        :param status: 状態
        :param actions_effective:  有効行動リスト
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
            q_max = self.__get_q(status, 0, actions_effective)
            action = 0
            for i in range(1, 4):
                q = self.__get_q(status, i, actions_effective)
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
        if experience is not None:
            # 学習データが存在する場合
            # 1回のプレイの経験のみを使用する
            # テーブルモードであれば複数プレイのデータを使用しても問題ないが,ニューラルネットワークモードの場合は
            # 手数と減衰率から行動価値Qの値がプレイによって大きく異なることにより,学習が進まない可能性があるため.
            experience = experience[0][0]
            q = 0
            count = len(experience['status'])
            if self.__mode_table:
                # テーブルモードの場合
                # 行動価値Qを蓄積するテーブルを作成
                q_total = np.zeros(self.__q_data.shape)
                q_delta_counter = np.zeros(self.__q_data.shape)
            else:
                # ニューラルネットワークモードの場合
                # 学習データを格納するリストを作成
                train_data = list()
                train_label = list()

            for i in range(count):
                # 新しい経験のデータから処理を実施
                status = experience['status'][-1 * i]
                actions_effective_one_hot = np.zeros([4])
                actions_effective_one_hot[experience['actions_effective'][-1 * i]] = 1
                action = experience['action'][-1 * i]
                # 行動価値Qを算出
                q = experience['reward'][-1 * i] + self.__decay * q
                if self.__mode_table:
                    # テーブルモードの場合
                    q_total[status[1], status[0], action] += q
                    q_delta_counter[status[1], status[0], action] += 1
                else:
                    # ニューラルネットワークモードの場合
                    is_register = False
                    for j in range(len(train_data)):
                        if train_data[j] == (status[1], status[0], action, actions_effective_one_hot[0], actions_effective_one_hot[1], actions_effective_one_hot[2], actions_effective_one_hot[3]):
                            # 登録済みのステータスと行動の組み合わせの場合
                            is_register = True
                            break

                    if not is_register:
                        # 未登録のステータスと行動の組み合わせの場合
                        train_data.append((status[1], status[0], action, actions_effective_one_hot[0], actions_effective_one_hot[1], actions_effective_one_hot[2], actions_effective_one_hot[3]))
                        train_label.append(q)

            if self.__mode_table:
                # テーブルモードの場合
                # 各状態,行動での行動価値Qの平均を算出
                q_total += self.__q_data * self.__q_count
                self.__q_count += q_delta_counter
                self.__q_data = q_total / self.__q_count

                # デーブルの各値をファイルに保存
                np.save('data\\monte_carlo\\q_data.npy', self.__q_data)
                np.save('data\\monte_carlo\\q_count.npy', self.__q_count)
            else:
                # ニューラルネットワークモードの場合
                # 学習を実施
                self.__model.fit(np.array(train_data), np.array(train_label), epochs=epochs)
                # 学習した重みをファイルに保存
                self.__model.save_weights('data\\monte_carlo\\weights.hdf5')

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
                        q_data[i, j, k] = self.__get_q((i, j), k, get_actions_effective((i, j)))

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
            actions_effective_one_hot = np.zeros([4])
            actions_effective_one_hot[actions_effective] = 1
            q = self.__model.predict(np.array([status[1], status[0], action, actions_effective_one_hot[0], actions_effective_one_hot[1], actions_effective_one_hot[2], actions_effective_one_hot[3]])[np.newaxis, :])[0][0]

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
