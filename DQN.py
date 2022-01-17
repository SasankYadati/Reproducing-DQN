import torch
import torchvision.transforms.functional as F
from collections import deque
import random
from network import CNN
from utils import plot, concate_observations, initialize_obs_q

torch.manual_seed(0)

class Transition:
    def __init__(self, curr_state, action, reward, next_state, is_terminal):
        self.curr_state = curr_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class DQNAgent:
    def __init__(self, num_actions, replay_memory, preprocess_fn, target_obs_size, eps_scheduler, batch_size):
        self.replay_memory = replay_memory
        
        self.q_network = CNN(num_actions).to('cuda')
        self.q_network_target = CNN(num_actions).to('cuda')
        self.q_network_target.eval()
        
        self.num_actions = num_actions
        
        self.last_state = None
        self.action = None
        self.curr_state = None
        
        self.step = 0
        self.eps_scheduler = eps_scheduler
        self.batch_size = batch_size
        
        self.preprocess_fn = preprocess_fn
        
        self.target_obs_size = target_obs_size
        self.latest_observations_preprocessed = initialize_obs_q(target_obs_size)
        
        self.criterion = torch.nn.HuberLoss().to('cuda')
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        
        self.GAMMA = 0.9
        self.UPDATE_TARGET_STEP = 250

    def policy(self, observation, reward, is_done):
        observation_processed = self.preprocess_fn(observation, self.target_obs_size)
        self.latest_observations_preprocessed.append(observation_processed)        
        next_state = concate_observations(self.latest_observations_preprocessed)
        # plot(next_state.permute(0,2,3,1))
        transition = Transition(self.curr_state, self.action, reward, next_state, is_done)
        if self.curr_state is not None:
            self.replay_memory.append(transition)
        self.curr_state = next_state.clone()
        epsilon = self.eps_scheduler.value(self.step)
        self.step += 1
        if self.step % self.UPDATE_TARGET_STEP == 0:
            self.q_network_target.load_state_dict(self.q_network.state_dict())
            self.q_network_target.eval()
        if random.random() <= epsilon:
            self.action = random.randint(0, self.num_actions-1)
        else:
            self.action = torch.argmax(self.q_network_target(self.curr_state.to('cuda')), 1)[0]            
        loss = 0.0
        if self.replay_memory.size >= self.batch_size:
            transistion_samples = self.replay_memory.sample(self.batch_size)
            loss = self.update_network(transistion_samples, self.q_network, self.GAMMA, self.criterion, self.optimizer, self.num_actions)
            self.step % self.UPDATE_TARGET_STEP == 0 and print(f"Step {self.step} loss {loss}")
        if is_done:
            self.initialize_obs_q()
        return self.action, epsilon


    def update_network(self, transitions, q_net, gamma, criterion, optimizer, num_actions):
        self.q_network.train()
        num_transistions = len(transitions)
        if num_transistions == 0:
            return None 
        inputs = torch.cat([t.curr_state for t in transitions]).to('cuda')
        targets = torch.zeros(num_transistions).to('cuda')
        one_hot_actions = torch.zeros((num_transistions, num_actions)).to('cuda')
        for i,t in enumerate(transitions):
            one_hot_actions[i][t.action] = 1
            if t.is_terminal:
                targets[i] = t.reward
            else:
                q_val_next = q_net(t.next_state.to('cuda'))
                q_val_next = torch.max(q_val_next).item()
                targets[i] = t.reward + gamma * q_val_next
        outputs = torch.sum(q_net(inputs) * one_hot_actions, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()
        return float(loss)
    
    def evaluate(self, states):
        self.q_network.eval()
        q_values = self.q_network(torch.cat(states).to('cuda'))
        q_values, _ = torch.max(q_values, 1)
        q_values = torch.mean(q_values)
        return q_values.item()