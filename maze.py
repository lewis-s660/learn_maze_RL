import numpy as np


class Maze:
    def __init__(self):
        """
        コンストラクタ
        """
        self.__position = np.zeros([2])
        self.__is_play = False
        self.__count = 0
        self.__wall_horizontal = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 1],
                                           [1, 1, 0, 1, 1, 0, 0, 1, 1],
                                           [1, 0, 0, 0, 0, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                           [1, 0, 1, 1, 0, 1, 0, 1, 1],
                                           [1, 0, 0, 1, 1, 0, 0, 1, 1],
                                           [1, 0, 0, 0, 0, 0, 1, 1, 1],
                                           [1, 0, 0, 0, 0, 1, 0, 0, 1]])
        self.__wall_vertical = np.array([[1, 0, 0, 0, 1, 0, 1, 0, 1],
                                         [1, 0, 1, 1, 0, 1, 1, 1, 1],
                                         [1, 1, 1, 0, 0, 0, 1, 1, 1],
                                         [1, 0, 0, 0, 0, 0, 0, 1, 1],
                                         [1, 1, 0, 0, 1, 0, 1, 1, 1],
                                         [1, 1, 1, 0, 0, 1, 1, 0, 1],
                                         [1, 1, 0, 0, 0, 1, 0, 1, 1],
                                         [1, 0, 0, 0, 0, 0, 0, 1, 1]])

    @property
    def status(self):
        """
        ステータス
        現在位置
        """
        return self.__position.copy()

    @property
    def is_play(self):
        """
        プレイ中フラグ
        True:プレイ中
        False:非プレイ中
        """
        return self.__is_play

    @property
    def score(self):
        """
        スコア
        設定しない(必ず0)
        """
        return 0

    @property
    def count(self):
        """手数"""
        return self.__count

    def start(self):
        """
        プレイを開始する
        :return:　なし
        """
        # 位置をクリア
        self.__position = np.zeros([2], dtype=np.int)

        # 手数をクリア
        self.__count = 0

        # プレイ中に変更
        self.__is_play = True

    def set_action(self, direction):
        """
        プレイヤーの位置を変更
        :param direction: 移動方向[0:上,1:右,2:下,3:左]
        :return: 配置成功フラグ(True:配置成功、False:配置失敗)
        """
        # 行動実施フラグ
        is_action = False

        if direction == 0:
            # 上方向に移動する場合
            if self.__wall_horizontal[self.__position[0], self.__position[1]] == 0:
                # 上方向に移動できる場合
                self.__position[1] -= 1
                is_action = True
        elif direction == 1:
            # 右方向に移動する場合
            if self.__wall_vertical[self.__position[1], self.__position[0] + 1] == 0:
                # 右方向に移動できる場合
                self.__position[0] += 1
                is_action = True
        elif direction == 2:
            # 下方向に移動する場合
            if self.__wall_horizontal[self.__position[0], self.__position[1] + 1] == 0:
                # 下方向に移動できる場合
                self.__position[1] += 1
                is_action = True
        elif direction == 3:
            # 左方向に移動する場合
            if self.__wall_vertical[self.__position[1], self.__position[0]] == 0:
                # 左方向に移動できる場合
                self.__position[0] -= 1
                is_action = True
        else:
            # 未定義の方向に移動する場合
            pass

        if is_action:
            self.__count += 1

        if (self.__position[0] == self.__wall_vertical.shape[0] - 1) \
                and (self.__position[1] == self.__wall_horizontal.shape[0] - 1):
            # ゴールした場合
            self.__is_play = False

        return is_action

    def get_actions_effective(self):
        """
        有効な行動リストを取得
        :return: 有効な行動リスト
        """
        actions = list()

        if self.__wall_horizontal[self.__position[0], self.__position[1]] == 0:
            # 上方向に移動できる場合
            actions.append(0)

        if self.__wall_vertical[self.__position[1], self.__position[0] + 1] == 0:
            # 右方向に移動できる場合
            actions.append(1)

        if self.__wall_horizontal[self.__position[0], self.__position[1] + 1] == 0:
            # 下方向に移動できる場合
            actions.append(2)

        if self.__wall_vertical[self.__position[1], self.__position[0]] == 0:
            # 左方向に移動できる場合
            actions.append(3)

        return actions

    def display(self, q_data=None):
        """
        表示出力
        :param q_data: 行動価値Qテーブル
        :return: なし
        """

        output = 'count:{0}\r\n'.format(self.__count)

        for i in range(self.__wall_horizontal.shape[1]):
            for j in range(self.__wall_horizontal.shape[0]):
                if q_data is None:
                    output += ' '
                    output += ' ' if self.__wall_horizontal[j, i] == 0 else '-'
                else:
                    output += (',' if self.__wall_horizontal[j, i] == 0 else '-,') * 4
            output += '\n\r'

            if i < self.__wall_vertical.shape[0]:
                if q_data is None:
                    for j in range(self.__wall_vertical.shape[1]):
                        output += ' ' if self.__wall_vertical[i, j] == 0 else '|'
                        if (self.__position[0] == j) and (self.__position[1] == i):
                            # プレイヤーが存在する位置の場合
                            output += '○'
                        elif (i == 0) and (j == 0):
                            # スタート地点の場合
                            output += 'S'
                        elif (i == self.__wall_horizontal.shape[0] - 1) and (j == self.__wall_vertical.shape[0] - 1):
                            # ゴール地点の場合
                            output += 'G'
                        else:
                            # 上記以外の地点の場合
                            output += ' '
                    output += '\n\r'
                else:
                    # 上方向の行動価値Qの値を出力するループ
                    for j in range(q_data.shape[1]):
                        output += '{0},,{1:.2f},,'.format(' ' if self.__wall_vertical[i, j] == 0 else '|', q_data[i, j, 0])
                    output += '{0}\r\n'.format(' ' if self.__wall_vertical[i, q_data.shape[1]] == 0 else '|')

                    # 左右方向の行動価値Qの値を出力するループ
                    for j in range(q_data.shape[1]):
                        point = ' '
                        if (self.__position[0] == j) and (self.__position[1] == i):
                            # プレイヤーが存在する位置の場合
                            point = '○'
                        elif (i == 0) and (j == 0):
                            # スタート地点の場合
                            point = 'S'
                        elif (i == self.__wall_horizontal.shape[0] - 1) and (j == self.__wall_vertical.shape[0] - 1):
                            # ゴール地点の場合
                            point = 'G'
                        else:
                            # 上記以外の地点の場合
                            point = ' '
                        output += '{0},{1:.2f},{2},{3:.2f},'.format(' ' if self.__wall_vertical[i, j] == 0 else '|',
                                                                    q_data[i, j, 3],
                                                                    point,
                                                                    q_data[i, j, 2])
                    output += '{0}\r\n'.format(' ' if self.__wall_vertical[i, q_data.shape[1]] == 0 else '|')

                    # 下方向の行動価値Qの値を出力するループ
                    for j in range(q_data.shape[1]):
                        output += '{0},,{1:.2f},,'.format(' ' if self.__wall_vertical[i, j] == 0 else '|', q_data[i, j, 2])
                    output += '{0}\r\n'.format(' ' if self.__wall_vertical[i, q_data.shape[1]] == 0 else '|')

        print(output)
