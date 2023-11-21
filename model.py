import os
import torch
import torch.nn as nn

class AtariNet(nn.Module):

    def __init__(self, nb_actions=4):

        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.action_value1 = nn.Linear(in_features=3136, out_features=1024)
        self.action_value2 = nn.Linear(in_features=1024, out_features=1024)
        self.action_value3 = nn.Linear(in_features=1024, out_features=nb_actions)

        self.state_value1 = nn.Linear(in_features=3136, out_features=1024)
        self.state_value2 = nn.Linear(in_features=1024, out_features=1024)
        self.state_value3 = nn.Linear(in_features=1024, out_features=1) 


    def forward(self, x):
        #print("PRITING X SHAPE")
        x = torch.Tensor(x)
        #print(x.shape)
        x = self.relu(self.conv1(x))
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        x = self.relu(self.conv3(x))
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)

        #print("PRITING STATE SHAPE")
        state_value = self.relu(self.state_value1(x))
        #print(state_value.shape)
        state_value = self.dropout(state_value)
        #print(state_value.shape)
        state_value = self.relu(self.state_value2(state_value))
        #print(state_value.shape)
        state_value = self.dropout(state_value)
        #print(state_value.shape)
        state_value = self.relu(self.state_value3(state_value))
        #print(state_value.shape)

        #print("PRITING ACTION SHAPE")
        action_value = self.relu(self.action_value1(x))
        #print(action_value.shape)
        action_value = self.dropout(action_value)
        #print(action_value.shape)
        action_value = self.relu(self.action_value2(action_value))
        #print(action_value.shape)
        action_value = self.dropout(action_value)
        #print(action_value.shape)
        action_value = self.state_value3(action_value)
        #print(action_value.shape)

        print("state_value shape:", state_value.shape, "action_value shape:", action_value.shape, "action_value mean:", action_value.mean())

        mean_value = action_value.mean(dim=1, keepdim=True)
        action_value_adjusted = action_value - mean_value
        output = state_value + action_value_adjusted

        return output
    
    def save_the_model(self,weights_filename='models/latest.pt'):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), weights_filename)


    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")

        except:
            print(f"No weights file available at {weights_filename}")