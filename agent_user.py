from agent_base import AgentBase


class AgentUser(AgentBase):
    def __init__(self):
        super().__init__()

    def get_action(self, status):
        while True:
            action = input('アクションを選択してください(0:上,1:右,2:下,3:左):')

            try:
                action = int(action)
                break
            except:
                pass

        return action
