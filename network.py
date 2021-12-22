import torch
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8,8), stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4,4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2592,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
            torch.nn.Softmax(1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    print(CNN(4)(torch.rand(16,4,84,84)))