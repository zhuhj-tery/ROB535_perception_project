import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import VehicleDataset


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)  # convolutional layer 1
        # print('Number of conv1 parameters: {}'.format(count_parameters(self.conv1)))
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)  # convolutional layer 2
        # print('Number of conv2 parameters: {}'.format(count_parameters(self.conv2)))
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)  # convolutional layer 3
        # print('Number of conv3 parameters: {}'.format(count_parameters(self.conv3)))
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2, padding=2)  # convolutional layer 4
        # print('Number of conv4 parameters: {}'.format(count_parameters(self.conv4)))
        self.fc1 = nn.Linear(1024, 64)  # fully connected layer 1
        # print('Number of fc1 parameters: {}'.format(count_parameters(self.fc1)))
        self.fc2 = nn.Linear(64,3)  # fully connected layer 2 (output layer)
        # print('Number of fc2 parameters: {}'.format(count_parameters(self.fc2)))

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / math.sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc1, self.fc2]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / math.sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        # print(x.shape)
        N, C, H, W = x.shape

        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = torch.flatten(z, 1) # Flatten z with start_dim=1
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = CNN()
    print(net)
    print('Number of CNN parameters: {}'.format(count_parameters(net)))
    dataset = VehicleDataset()
    images, labels = iter(dataset.train_loader).next()
    print('Size of model output:', net(images).size())
