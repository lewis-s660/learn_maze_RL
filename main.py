from datetime import datetime
from statistics import mean


from control import Control
from maze import Maze
from agent_user import AgentUser
from agent_random import AgentRandom
from agent_monte_carlo import AgentMonteCarlo
from agent_dynamic_programing import AgentDynamicPrograming
from agent_td_q import AgentTDQ

# モードを指定
mode = 'dynamic_programing'

mode_table = False
# プレイ回数
count_play = 1
# 最大ループ数
count_loop_max = 1000
# エージェント切り替え境界値
boundary_change_agent = 0
# ループでの表示間隔
step_indicate = 100
# エポック数
epochs = 10000

# 環境を生成
environment = Maze()

# モードに合わせたエージェントを生成
agent_1 = None
agent_2 = None
if mode == 'User':
    # ユーザーモードの場合
    agent_1 = AgentUser()
elif mode == 'Random':
    # ランダムモードの場合
    agent_1 = AgentRandom()
elif mode == 'monte_carlo':
    # モンテカルロ法モードの場合
    if mode_table:
        # テーブルモードの場合
        agent_1 = AgentMonteCarlo(mode_table=mode_table, decay=0.99, count_random_policy=count_loop_max)
        # プレイ回数
        count_play = 1
        # 最大ループ数
        count_loop_max = 1000
    else:
        # ニューラルネットワークモードの場合
        agent_1 = AgentMonteCarlo(mode_table=mode_table, decay=0.99, count_random_policy=1)
        #agent_2 = AgentMonteCarlo(mode_table=False)
elif mode == 'dynamic_programing':
    # 動的計画法モードの場合
    if mode_table:
        # テーブルモードの場合
        agent_1 = AgentDynamicPrograming(environment=environment, mode_table=mode_table)
        # プレイ回数を0に変更
        count_play = 0
        # 最大ループ数を1に変更(実際にプレイする必要がないため1回の学習(エポック数は1ではない)でよい)
        count_loop_max = 1000
        # エポック数を変更
        epochs = 100000
    else:
        # ニューラルネットワークモードの場合
        agent_1 = AgentDynamicPrograming(environment=environment, mode_table=mode_table)
        # プレイ回数を0に変更
        count_play = 0
        # 最大ループ数を1に変更(実際にプレイする必要がないため1回の学習(エポック数は1ではない)でよい)
        count_loop_max = 1000
        # エポック数を変更
        epochs = 10000
elif mode == 'td_q':
    # 動的計画法モードの場合
    if mode_table:
        # テーブルモードの場合
        agent_1 = AgentTDQ(environment=environment, mode_table=mode_table)
    else:
        # ニューラルネットワークモードの場合
        agent_1 = AgentTDQ(environment=environment, mode_table=mode_table)

# 制御インスタンスを生成
control_1 = Control(environment, [agent_1], is_display=False)
control_2 = None
if agent_2 is not None:
    # 2つ目のエージェントが生成されている場合
    control_2 = Control(environment, [agent_2], is_display=False)

count_to_goal = list()

# 指定プレイ回数のプレイと学習のセットを指定回数ループ
for i in range(count_loop_max):
    experience = None
    if 0 < count_play:
        # プレスする場合
        if (i <= boundary_change_agent) or control_2 is None:
            # エージェント切り替え境界値以下または2つ目の制御インスタンスが存在しない場合
            control = control_1
        else:
            # エージェント切り替え境界値超過かつ2つ目の制御インスタンスが存在する場合
            control = control_2

        # 指定回数のプレイを実施
        experience = control.play(count_play, is_indicate=True)
        # ゴールまでの手数を記憶
        count_to_goal.append(environment.count)

        if (0 < i) and ((i % step_indicate == 0) or (i == count_loop_max - 1)):
            # 表示のタイミングの場合
            print('プレイ回数：{0} 攻略手数：{1} 過去{2}回の平均：{3:.2f} 過去{2}回の最小攻略手数:{4}'.format(i, environment.count, step_indicate, mean(count_to_goal[-100:]), min(count_to_goal[-100:])))

    if control_2 is None:
        # 2つ目の制御インスタンスが存在しない場合
        agent = agent_1
    else:
        # 2つ目の制御インスタンスが存在する場合
        agent = agent_2

    # 学習を実施
    agent.fit(experience, number=i, epochs=epochs)
    # 学習データを表示
    #environment.display(agent.get_q_table_experience(experience))

try:
    # 学習後の行動価値Qの値を出力
    environment.display(agent.get_q_table(environment.get_actions_effective))
except:
    # 学習後の行動価値Qの値を出力
    environment.display(agent.get_v_table(), is_q=False)


