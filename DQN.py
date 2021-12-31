from typing import Deque
import torch
import torchvision.transforms.functional as F
from collections import deque
import random
from network import CNN

class Transition:
    def __init__(self, curr_state, action, reward, next_state, is_terminal):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class DQNAgent:
    def __init__(self, num_actions):
        self.replay_memory = deque()
        self.MEMORY_CAPCITY = 1000000
        self.q_network = CNN(num_actions).to('cuda')
        self.q_network_target = CNN(num_actions).to('cuda')
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
        self.GAMMA = 0.9
        self.UPDATE_TARGET_STEP = 250
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters())

    def get_epsilon_using_linear_annealing(step, max_steps, min_epsilon, initial_epsilon):
        if step > max_steps:
            return min_epsilon
        return ((min_epsilon - initial_epsilon) * (step) / (max_steps)) + initial_epsilon
    
    def update_memory(self, transition):
        if len(self.replay_memory) == self.MEMORY_CAPCITY:
            self.replay_memory.popleft()
        self.replay_memory.append(transition)

    def policy(self, observation, reward, is_done):
        # self.last_state = self.curr_state
        self.latest_observations_preprocessed.popleft()
        observation_processed = DQNAgent.preprocess(observation)
        self.latest_observations_preprocessed.append(observation_processed)        
        next_state = DQNAgent.concate_observations(self.latest_observations_preprocessed, 4)
        transition = Transition(self.curr_state, self.action, reward, next_state, is_done)
        if self.curr_state is not None:
            self.update_memory(transition)
        self.curr_state = next_state
        epsilon = DQNAgent.get_epsilon_using_linear_annealing(self.step, self.MAX_ANNEALING_STEPS, self.EPSILON_MIN, self.EPSILON_INITIAL)
        self.step += 1
        if self.step % self.UPDATE_TARGET_STEP == 0:
            self.q_network_target.load_state_dict(self.q_network.state_dict())
        if random.random() <= epsilon:
            self.action = random.randint(0, self.num_actions-1)
        else:
            self.action = torch.argmax(self.q_network_target(self.curr_state.to('cuda')), 1)[0]            
        loss = 0.0
        if len(self.replay_memory) >= self.BATCH_SIZE:
            loss = self.update_network(DQNAgent.sample_transistions_from_experience(self.replay_memory, self.BATCH_SIZE), self.q_network, self.GAMMA, self.criterion, self.optimizer, self.num_actions)
            if self.step % 250 == 0:
                print(f"Step {self.step} loss {loss}")
        return self.action, epsilon


    def update_network(self, transitions, q_net, gamma, criterion, optimizer, num_actions):
        num_transistions = len(transitions)
        inputs = torch.cat([t.curr_state for t in transitions]).view(num_transistions, 4, 84, 84).to('cuda')
        targets = torch.cat([torch.zeros((1,num_actions)) for _ in transitions]).view(num_transistions, num_actions).to('cuda')
        for i,t in enumerate(transitions):
            if t.is_terminal:
                targets[i][t.action] = t.reward
            else:
                q_val_next = torch.max(q_net(t.next_state.to('cuda'))).item()
                targets[i][t.action] = t.reward + gamma * q_val_next
        outputs = q_net(inputs)
        optimizer.zero_grad()
        criterion.to('cuda')
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

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
    for i in range(1000000):
        eps = DQNAgent.get_epsilon_using_linear_annealing(i, 1000000, 0.1, 1.0)
        i % 100 == 0 and print(i, eps)
