import os
from breakout import DQNBreakout

import gym
import numpy as np
from PIL import Image
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

environment = DQNBreakout(device = device, render_mode='human')

state = environment.reset()

for i in range(10):
    environment.reset()
    done = False
    while not done:
        action = environment.action_space.sample()
        state, reward, done, info = environment.step(action)
        environment.render()


