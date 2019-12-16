from datetime import datetime
from statistics import mean


from control import Control
from maze import Maze
from agent_user import AgentUser
from agent_random import AgentRandom
from agent_monte_carlo import AgentMonteCarlo

size_board = (9, 9)
name = 'pattern_1'
path = 'weights'

environment = Maze()

#agent = AgentUser()
#agent = AgentRandom()
#agent = AgentMonteCarlo(epsilon=0.001)
agent = AgentMonteCarlo(mode_table=True)
agent_2 = AgentMonteCarlo(mode_table=False)

control = Control(environment, [agent], is_display=False)
control_2 = Control(environment, [agent_2], is_display=False)

count = list()

for i in range(10000):
    if i < 10:
        experience = control.play(1, is_indicate=True)
    else:
        experience = control_2.play(1, is_indicate=True)

    count.append(environment.count)
    if i % 100 == 0:
        print('プレイ回数：{0} 攻略手数：{1} 過去100回の平均：{2:.2f} 過去100回の最小攻略手数:{3}'.format(i, environment.count, mean(count[-100:]), min(count[-100:])))

    agent_2.fit(experience, number=i, epochs=10000)

print(mean(count))



