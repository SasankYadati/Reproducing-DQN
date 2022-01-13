import gym
from dqn import DQNAgent
from network import CNN

env = gym.make('Breakout-v4')

q_net = CNN(env.action_space.n)
agent = DQNAgent(env.action_space.n)
SKIP_FRAMES = 4
NUM_EPISODES = 100
frames_seen = 0
frames = 0
eps_rewards = dict()
total_reward = 0.0
exp_mov_avg = 0.0
for i in range(NUM_EPISODES):
    done = False
    observation = env.reset()
    eps_reward = 0
    reward, done = 0, False
    while not done:    
        action, eps = agent.policy(observation, reward, done)
        frames_seen += 1
        for _ in range(SKIP_FRAMES):
            frames += 1
            observation, reward, done, _ = env.step(action)
            eps_reward += reward
            if done:
                break
    # eps_rewards[i] = eps_reward
    exp_mov_avg = 0.9 * exp_mov_avg + (0.1 * eps_reward)
    total_reward += eps_reward
    print(f"Episode {i} Epsilon = {eps:.5f} Reward = {eps_reward:.3f} Avg Reward = {total_reward/(i+1):.3f} Exp Moving Avg = {exp_mov_avg:.3f}")
    if (eps_reward) >= 20.0:
        break