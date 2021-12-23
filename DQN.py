import torch
import torchvision.transforms.functional as F
from collections import dequeue

class Transistion:
    def __init__(self, curr_state, action, reward, next_state):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class DQN:
    def __init__(self, q_network, num_actions):
        self.replay_memory = []
        self.q_network = q_network
        self.latest_observations_preprocessed = dequeue([torch.zeros(1,84,84),torch.zeros(1,84,84),torch.zeros(1,84,84),torch.zeros(1,84,84)])
        self.num_actions = num_actions
        self.last_state = None
        self.curr_state = None

    def policy(self, observation, reward):
        self.latest_observations_preprocessed.popleft()
        observation_processed = DQN.preprocess(observation)
        self.latest_observations_preprocessed.append(observation_processed)
        
        self.last_state = self.curr_state
        self.curr_state = DQN.concate_latest_observations(self.latest_observations_preprocessed, 4)
        transistion = Transistion(self.last_state, self.action, reward, self.curr_state)
        if self.last_state is not None:
            self.replay_memory.append(transistion)        
        
        self.action = self.q_network(self.curr_state)
        return self.action    

    def preprocess(observation:torch.Tensor):
        observation = torch.Tensor(observation).view(3, 210, 160).type(torch.uint8)
        observation = F.rgb_to_grayscale(observation)
        observation = F.resize(observation, size=(110, 84))
        observation = F.crop(observation, 0, 0, 84, 84)
        return observation

    def concate_observations(q:dequeue, n:int):
        assert n <= len(q)
        return torch.cat([q[x] for x in range(n)])

if __name__ == '__main__':
    import gym
    import network
    env = gym.make('Breakout-v4', render_mode='human')
    q_net = network.CNN(env.action_space.n)
    dqn = DQN(q_net, env.action_space.n) 
