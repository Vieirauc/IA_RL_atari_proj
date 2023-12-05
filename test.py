import os
from breakout import DQNBreakout

import gym
import numpy as np
from PIL import Image
import torch
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

environment = DQNBreakout(device = device, render_mode='human')

model = AtariNet(nb_actions=4)

model.to(device)

model.load_the_model(weights_filename='models/latest.pt')

agent = Agent(model=model,
              device=device,
              epsilon=0.05,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.00001,
              memory_capacity=1000000,
              batch_size=64)

agent.test(env=environment)



'''
state = environment.reset()

action_probs = model.forward(state).detach()
print(f"{action_probs}, {torch.agrmax(action_probs, dim=-1, keepdim=True)}")
'''

'''
for i in range(10):
    environment.reset()
    done = False
    while not done:
        action = environment.action_space.sample()
        state, reward, done, info = environment.step(action)
        environment.render()
    
    print(state.shape)

'''

