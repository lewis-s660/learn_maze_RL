import numpy as np
import tensorflow as tf

from agent_base import AgentBase


class AgentDynamicPrograming(AgentBase):
    def __init__(self, environment, epsilon=0.1, decay=0.9, eta=0.1, gradient_minimum=0.001, mode_table=True, size=(8, 8)):
        """
        コンストラクタ
        :param epsilon: ε-Greedy方策で使用するεの値
        :param decay: 行動価値Qを算出する際の減衰率
        :param eta: 学習率
        :param gradient_minimum: 学習が必要となる最小の勾配
        :param mode_table: テーブルモード選択フラグ(True:テーブルモード,False:ニューラルネットワークモード)
        :param size: 状態サイズ
        """
        super().__init__()
        self.__environment = environment
        self.__epsilon = epsilon
        self.__decay = decay
        self.__eta = eta
        self.__gradient_minimum = gradient_minimum
        self.__mode_table = mode_table
        self.__size = np.array(size)
        self.__count_random_policy = 0

        if self.__mode_table:
            # テーブルモードの場合
            try:
                # テーブルの情報をファイルから読み込み
                self.__v_data = np.load('data\\dynamic_programing\\v_data.npy')
            except:
                # ファイルからの読み込みに失敗した場合はすべて0で領域を確保
                self.__v_data = np.zeros([self.__size[0], self.__size[1]])
        else:
            # ニューラルネットワークモードの場合
            try:
                # モデルをファイルから読み込み
                self.__model = tf.keras.models.load_model('data\\dynamic_programing\\model.hdf5')
            except:
                # モデルの読み込みに失敗した場合はモデルを生成
                self.__model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, input_shape=(2, ), activation='relu'),
                                                           tf.keras.layers.Dense(1024, activation='relu'),
                                                           tf.keras.layers.Dense(64, activation='relu'),
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
            reward = -10
        if not is_play:
            # ゴールした場合
            reward = 1000
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

        train_data = list()
        train_label = list()

        # 学習を開始
        for i in range(epochs):
            # とりうる位置を1次元配列で取得
            indices = np.random.choice(range(self.__size[0] * self.__size[1]),
                                       self.__size[0] * self.__size[1],
                                       replace=False)

            # 1つ以上の勾配の傾きが最小勾配より大きいかどうかのフラグ
            greater_gradient = False
            for index in indices:
                # スカラー値を座標に変換
                status = np.array([index // self.__size[0], index % self.__size[0]])
                # 有効な行動リストを取得
                actions = self.__environment.get_actions_effective(status)

                v = 0
                count = 0
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

                    if action in actions:
                        # 行動が有効行動リストに含まれる場合
                        if (status_next == np.array([7, 7])).all():
                            # 行動有効かつ非プレイ中として報酬を取得
                            reward = self.get_reward(None, None, True, None, False, None, actions_effective_next=actions)
                        else:
                            # 行動有効かつプレイ中として報酬を取得
                            reward = self.get_reward(None, None, True, None, True, None, actions_effective_next=actions)
                        # 価値Vの値を算出
                        v += reward + self.__decay * self.__get_v(status_next)
                        count += 1

                # 価値Vを取得
                data = self.__get_v(status)
                # 勾配を算出
                gradient = (v / count) - data
                if self.__gradient_minimum < abs(gradient):
                    # 勾配の傾きが最小勾配より大きい場合
                    greater_gradient = True

                if self.__mode_table:
                    # テーブルモードの場合
                    self.__v_data[status[1], status[0]] = data + self.__eta * gradient
                else:
                    # ニューラルネットワークモードの場合
                    train_data.append(status)
                    train_label.append(v / count)

            if not self.__mode_table:
                # ニューラルネットワークモードの場合
                # 1回のデータ収集で処理を抜ける(テーブルモードとニューラルネットワークモードでのエポック数の概念の違いによる)
                break

            if (i + 1) % 100 == 0:
                print('ループ数：{0}  エポック数：{1} / {2}'.format(number, i + 1, epochs))

            if not greater_gradient:
                # すべての勾配の傾きが最小勾配以下の場合
                print('ループ数：{0}  エポック数：{1} / {2}'.format(number, i + 1, epochs))
                break

        if self.__mode_table:
            # テーブルモードの場合
            # デーブルの各値をファイルに保存
            np.save('data\\dynamic_programing\\v_data.npy', self.__v_data)
        else:
            # ニューラルネットワークモードの場合
            print('{0}回目の学習'.format(number + 1))
            if greater_gradient:
                # 1つ以上の勾配の傾きが最小勾配より大きい場合
                # 学習を実施
                self.__model.fit(np.array(train_data), np.array(train_label), epochs=epochs, verbose=0)
                # 学習した重みをファイルに保存
                self.__model.save_weights('data\\dynamic_programing\\weights.hdf5')

    def get_v_table(self):
        """
        価値Vのテーブル取得処理
        価値Vのテーブルを返す(テーブルがない場合は生成も行う)
        :return: 価値Vテーブル(x座標, y座標)
        """
        # 戻り値用の領域をすべて0で生成
        v_data = np.zeros([self.__size[0], self.__size[1]])
        if self.__mode_table:
            # テーブルモードの場合
            v_data = self.__v_data.copy()
        else:
            # ニューラルネットワークモードの場合
            for i in range(v_data.shape[0]):
                for j in range(v_data.shape[1]):
                    v_data[i, j] = self.__get_v((i, j))

        return v_data

    def __get_v(self, status_next):
        """
        価値V取得処理
        次の状態の価値Vを返す
        :param status_next: 次の状態
        :return: 価値V
        """
        v = 0
        if self.__mode_table:
            # テーブルモードの場合
            v = self.__v_data[status_next[1], status_next[0]]
        else:
            # ニューラルネットワークモードの場合
            v = self.__model.predict(np.array([status_next[1], status_next[0]])[np.newaxis, :])[0][0]

        return v
