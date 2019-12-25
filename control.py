from datetime import datetime, timedelta


class Control:
    def __init__(self, environment, players, is_display=True):
        self.__environment = environment
        self.__players = players
        self.__is_display = is_display
        self.__time_start = datetime.now()
        self.__time_elapsed = timedelta()

    @property
    def time_elapsed(self):
        """
        経過時間
        :return: 経過時間
        """
        if self.environment.is_play:
            # プレイ中の場合
            # 経過時間を更新
            self.__time_elapsed = datetime.now() - self.__time_start

        return self.__time_elapsed

    def play(self, count=1, is_indicate=True, step_max=0):
        """
        プレイを実施
        :param count: プレイ回数
        :param is_indicate: 表示フラグ
        :param step_max: 最大ステップ数
        :return: プレイを実施しての経験
        """

        step = step_max

        # 経験(1階層：各プレイヤーのリスト、2階層：各プレイ情報のリスト、3階層：属性のディクショナリ、4階層：データの値)
        experience = list()
        for j in range(len(self.__players)):
            experience.append(list())

        for i in range(count):
            if is_indicate:
                print('start play:{0}回目'.format(i + 1))
            # 各種データの初期化
            self.__time_start = datetime.now()
            self.__environment.start()
            for j in range(len(self.__players)):
                self.__players[j].initialize()
                experience[j].append(dict())
                experience[j][-1]['status'] = list()
                experience[j][-1]['actions_effective'] = list()
                experience[j][-1]['action'] = list()
                experience[j][-1]['status_next'] = list()
                experience[j][-1]['action_next'] = list()
                experience[j][-1]['actions_effective_next'] = list()
                experience[j][-1]['reward'] = list()
                experience[j][-1]['q'] = list()

            if self.__is_display:
                # 表示する場合
                # 初期状態を表示
                self.__environment.display()

            counter = 0
            is_first = [True] * len(self.__players)

            # プレイを最後まで実施
            while True:
                for j in range(len(self.__players)):
                    while True:
                        counter += 1
                        # 行動前の状態を取得
                        status = self.__environment.status
                        # 行動前の有効な行動リストを取得
                        actions_effective = self.__environment.get_actions_effective()
                        # 行動を取得
                        if is_first[j] or not self.__players[j].mode_sarsa:
                            # 初回取得またはSARSAモードでない場合
                            action = self.__players[j].get_action(self.__environment.status, actions_effective)
                        else:
                            # SARSAモードの場合
                            action = self.__players[j].get_action(self.__environment.status, actions_effective, is_previous=False)
                        is_first[j] = False
                        # 行動を実施
                        can_action = self.__environment.set_action(action)
                        # 行動後の有効な行動リストを取得
                        actions_effective_next = self.__environment.get_actions_effective()
                        # 環境からの情報を通知
                        reward = self.__players[j].get_reward(status,
                                                              action,
                                                              can_action,
                                                              self.__environment.status,
                                                              self.__environment.is_play,
                                                              self.__environment.score,
                                                              actions_effective_next)
                        action_next = None
                        if self.__players[j].mode_sarsa:
                            # SARSAモードの場合
                            # 次回の行動を取得する
                            action_next = self.__players[j].get_action(self.__environment.status, actions_effective)
                        q = self.__players[i].get_q(status, action, self.__environment.status, action_next, reward)

                        experience[j][-1]['status'].append(status)
                        experience[j][-1]['actions_effective'].append(actions_effective)
                        experience[j][-1]['action'].append(action)
                        experience[j][-1]['status_next'].append(self.__environment.status)
                        experience[j][-1]['action_next'].append(action_next)
                        experience[j][-1]['actions_effective_next'].append(actions_effective_next)
                        experience[j][-1]['reward'].append(reward)
                        experience[j][-1]['q'].append(q)

                        if is_indicate and (counter % 100 == 0):
                            print('　　　play count:{0}回目'.format(counter))

                        if can_action:
                            # 行動が実行できた場合
                            break

                    if self.__is_display:
                        # 表示する場合
                        # 初期状態を表示
                        self.__environment.display()

                    if not self.__environment.is_play:
                        # プレイが終了していた場合
                        # 経過時間の計測を停止
                        self.__time_elapsed = datetime.now() - self.__time_start
                        break

                if not self.__environment.is_play:
                    # プレイが終了していた場合
                    # 経験を調整
                    for j in range(len(self.__players)):
                        self.__players[j].adjust_experience(experience[j][-1], self.__environment.score)
                    break

                if 0 < step_max:
                    # 最大ステップ数の指定がある場合
                    step -= 1
                    if step <= 0:
                        # 指定回数のステップの実行が終わった場合
                        break

            # 終了処理を実施
            for j in range(len(self.__players)):
                self.__players[j].finalize(self.__environment.status, self.__environment.score)

        return experience
