import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNet(nn.Module):
    def __init__(self, nb_actions=4):
        super(AtariNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, nb_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))

        return self.fc2(x)

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, 1, 84, 84)))).view(1, -1).size(1)

    def save_the_model(self, weights_filename='models/latest.pt'):
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)


    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")