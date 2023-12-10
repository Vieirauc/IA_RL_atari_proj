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

input_file = 'latest.pt'

model = AtariNet(nb_actions=4)

model.load_the_model(weights_filename=input_file)

agent = Agent(model=model,
              device=device,
              epsilon=0.05,
              nb_warmup=50,
              nb_actions=4,
              memory_capacity=1000000,
              batch_size=64)

test_environment = DQNBreakout(device = device, render_mode='human')

agent.test(env=test_environment)


