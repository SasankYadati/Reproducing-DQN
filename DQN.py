import torch
import torchvision.transforms.functional as F
from collections import deque
import random

class Transition:
    def __init__(self, curr_state, action, reward, next_state, is_terminal):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class DQN:
    def __init__(self, q_network, num_actions):
        self.replay_memory = []
        self.q_network = q_network
        self.latest_observations_preprocessed = deque(
            [
                torch.zeros(1,84,84),
                torch.zeros(1,84,84),
                torch.zeros(1,84,84),
                torch.zeros(1,84,84)
            ]
        )
        self.num_actions = num_actions
        self.last_state = None
        self.action = None
        self.curr_state = None
        self.step = 0
        self.EPSILON_MIN = 0.1
        self.EPSILON_INITIAL = 1.0
        self.MAX_ANNEALING_STEPS = 1000000
        self.BATCH_SIZE = 32

    def get_epsilon_using_linear_annealing(step, max_steps, min_epsilon, initial_epsilon):
        if step > max_steps:
            return min_epsilon
        return ((min_epsilon - initial_epsilon) * (step) / (max_steps)) + initial_epsilon

    def policy(self, observation, reward, is_done):
        self.last_state = self.curr_state
        self.latest_observations_preprocessed.popleft()
        observation_processed = DQN.preprocess(observation)
        self.latest_observations_preprocessed.append(observation_processed)        
        self.curr_state = DQN.concate_observations(self.latest_observations_preprocessed, 4)
        transition = Transition(self.last_state, self.action, reward, self.curr_state, is_done)
        self.last_state is not None and self.replay_memory.append(transition)
        epsilon = DQN.get_epsilon_using_linear_annealing(self.step, self.MAX_ANNEALING_STEPS, self.EPSILON_MIN, self.EPSILON_INITIAL)
        self.step += 1
        if random.random() <= epsilon:
            self.action = random.randint(0, self.num_actions-1)
        else:
            self.action = torch.argmax(self.q_network(self.curr_state), 1)[0]            

        # self.update_network(DQN.sample_transistions_from_experience(self.replay_memory, self.BATCH_SIZE))
        return self.action


    def update_network(self, transitions):
        return        

    def preprocess(observation:torch.Tensor):
        observation = torch.Tensor(observation).view(3, 210, 160).type(torch.float32)
        observation = F.rgb_to_grayscale(observation)
        observation = F.resize(observation, size=(110, 84))
        observation = F.crop(observation, 0, 0, 84, 84)
        return observation

    def concate_observations(q:deque, n:int):
        assert n <= len(q)
        state = torch.unsqueeze(torch.cat([q[x] for x in range(n)]), 0)
        return state

    def sample_transistions_from_experience(replay_memory, num_transistions):
        return random.sample(replay_memory, num_transistions)

if __name__ == '__main__':
    eps = 1.0
    for i in range(1000):
        eps = DQN.get_epsilon_using_linear_annealing(i, 1000, 0.1, 1.0)
        print(eps)
