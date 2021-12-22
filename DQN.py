class Transistion:
    def __init__(self, curr_state, action, reward, next_state):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class DQN:
    def __init__(self, q_network, num_actions, obs_space):
        self.replay_memory = []
        self.q_network = q_network

    def action(self):
        pass

    def preprocess(self, state):
        pass

    def store_transistion(self, transistion:Transistion):
        self.replay_memory.append(transistion)



if __name__ == '__main__':
    import network
    import gym
    env = gym.make('Breakout-v4', render_mode='human')
    q_net = network.CNN(env.action_space.n)
    dqn = DQN(q_net, env.action_space.n, env.observation_space) 
