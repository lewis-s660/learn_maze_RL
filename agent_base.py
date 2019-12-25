import numpy as np


class AgentBase:
    def __init__(self):
        pass

    def initialize(self):
        """
        初期化実行処理
        各プレイの開始前の初期化処理を実施する.
        :return: なし
        """
        pass

    def finalize(self, status, score):
        """
        終了処理
        各プレイの終了後の処理を実施する
        :param status: 状態
        :param score: スコア
        :return: なし
        """
        pass

    def get_action(self, status, actions_effective, is_previous=False):
        """
        行動取得処理
        状態から行動を決定して返す
        :param status: 状態
        :param actions_effective:  有効行動リスト
        :param is_previous: 前回取得値取得フラグ
        :return: 行動
        """
        return 0

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
        return 0

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
        return 0

    def adjust_experience(self, experience, score):
        """
        経験の調整処理
        一連のプレイの後に経験の情報を調整する
        :param experience: 経験
        :param score: スコア
        :return: なし
        """
        pass

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
        pass

    def get_q_table(self, get_actions_effective):
        """
        行動価値Qのテーブル取得処理
        行動価値Qのテーブルを返す(テーブルがない場合は生成も行う)
        :param get_actions_effective: 有効行動リスト取得ハンドラ
        :return: 行動価値Qテーブル(x座標, y座標, 行動)
        """
        return np.array([[[0]]])


    def get_v_table(self):
        """
        価値Vのテーブル取得処理
        価値Vのテーブルを返す(テーブルがない場合は生成も行う)
        :return: 価値Vテーブル(x座標, y座標)
        """
        return np.array([[[0]]])