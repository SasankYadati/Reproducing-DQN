import torch.nn as nn
import torch
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        # self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # self.batch_norm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(3456, 256)
        self.fc2 = nn.Linear(256, num_classes)
        conv_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        for layer in conv_layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

if __name__ == '__main__':
    print(CNN(4)(torch.rand(16,4,110,84)))
    # MEMORY_CAPCITY = 10
    # replay_memory = ReplayMemory(10)
    # for i in range(50):
    #     replay_memory.append(i)
    # print(replay_memory)
