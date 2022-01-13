from typing import Deque
import torch
import torchvision.transforms.functional as F
from collections import deque
import random
from network import CNN
from torch.profiler import profile, record_function, ProfilerActivity

class Transition:
    def __init__(self, curr_state, action, reward, next_state, is_terminal):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        old_obj = self.buffer[self.index]
        del old_obj
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

class DQNAgent:
    def __init__(self, num_actions):
        self.MEMORY_CAPCITY = 50000
        self.replay_memory = ReplayMemory(self.MEMORY_CAPCITY)
        self.q_network = CNN(num_actions).to('cuda')
        self.q_network_target = CNN(num_actions).to('cuda')
        self.latest_observations_preprocessed = deque(
            [
                torch.zeros(1,110,84),
                torch.zeros(1,110,84),
                torch.zeros(1,110,84),
                torch.zeros(1,110,84)
            ],
            maxlen=4
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
        self.UPDATE_TARGET_STEP = 500
        self.criterion = torch.nn.MSELoss().to('cuda')
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters())

    def get_epsilon_using_linear_annealing(step, max_steps, min_epsilon, initial_epsilon):
        if step > max_steps:
            return min_epsilon
        return ((min_epsilon - initial_epsilon) * (step) / (max_steps)) + initial_epsilon

    def policy(self, observation, reward, is_done):
        observation_processed = DQNAgent.preprocess(observation)
        self.latest_observations_preprocessed.append(observation_processed)        
        next_state = DQNAgent.concate_observations(self.latest_observations_preprocessed, 4)
        transition = Transition(self.curr_state, self.action, reward, next_state, is_done)
        if self.curr_state is not None:
            self.replay_memory.append(transition)
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
        if self.replay_memory.size >= self.BATCH_SIZE:
            transistion_samples = self.replay_memory.sample(self.BATCH_SIZE)
            loss = self.update_network(transistion_samples, self.q_network, self.GAMMA, self.criterion, self.optimizer, self.num_actions)
            self.step % 250 == 0 and print(f"Step {self.step} loss {loss}")                
        return self.action, epsilon


    def update_network(self, transitions, q_net, gamma, criterion, optimizer, num_actions):
        num_transistions = len(transitions)
        if num_transistions == 0:
            return None 
        inputs = torch.cat([t.curr_state for t in transitions]).reshape(num_transistions, 4, 110, 84).to('cuda')
        targets = torch.cat([torch.zeros((1,num_actions)) for _ in transitions]).reshape(num_transistions, num_actions).to('cuda')
        for i,t in enumerate(transitions):
            if t.is_terminal:
                targets[i][t.action] = t.reward
            else:
                q_val_next = torch.max(q_net(t.next_state.to('cuda'))).item()
                targets[i][t.action] = t.reward + gamma * q_val_next
        outputs = q_net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()
        return float(loss)

    def preprocess(observation:torch.Tensor):
        observation = torch.Tensor(observation).reshape(3, 210, 160)
        observation = F.rgb_to_grayscale(observation)
        observation = F.resize(observation, size=(110, 84))
        return observation

    def concate_observations(q:deque, n:int):
        assert n <= len(q)
        state = torch.unsqueeze(torch.cat([q[x] for x in range(n)]), 0)
        return state

if __name__ == '__main__':
    # eps = 1.0
    # for i in range(1000000):
    #     eps = DQNAgent.get_epsilon_using_linear_annealing(i, 1000000, 0.01, 1.0)
    #     i % 100 == 0 and print(i, eps)
    mem = ReplayMemory(10)
    for i in range(100):
        if mem.size > 4:
            print(mem.buffer)
        mem.append(i)
