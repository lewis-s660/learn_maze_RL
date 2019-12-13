from datetime import datetime


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
agent = AgentMonteCarlo()

control = Control(environment, [agent], is_display=False)

count = list()

for i in range(100000):
    experience = control.play(1)
    count.append(environment.count)
    print('プレイ回数：{0} 攻略手数：{1}'.format(i, environment.count))

    agent.fit(experience, number=i)

print(count)
print(min(count))


