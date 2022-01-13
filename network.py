import torch.nn as nn
import torch
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4480, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

if __name__ == '__main__':
    print(CNN(4)(torch.rand(16,4,110,84)))
    # MEMORY_CAPCITY = 10
    # replay_memory = ReplayMemory(10)
    # for i in range(50):
    #     replay_memory.append(i)
    # print(replay_memory)
