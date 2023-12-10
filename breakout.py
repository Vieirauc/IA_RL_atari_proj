import collections
#import cv2
import gym
import numpy as np
from PIL import Image
import torch

class DQNBreakout(gym.Wrapper):

    def __init__(self, render_mode='rgb_array', repeat = 4, device = 'cpu', no_ops = 1, fire_first = False):
        env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)
        
        super(DQNBreakout, self).__init__(env)

        self.image_shape = (84, 84)
        self.frame_buffer = collections.deque(maxlen=4)
        self.no_ops = 0
        self.fire_first = fire_first
        self.repeat = repeat
        self.device = device
        self.lives = env.ale.lives()


    def step(self, action):

        total_reward = 0
        done = False

        for i in range(self.repeat):
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            current_lives = info['lives']

            if current_lives < self.lives:
                total_reward = total_reward - 1
                self.lives = current_lives

            self.frame_buffer.append(observation)

            if done:
                break

        max_frame = np.max(np.stack(self.frame_buffer), axis=0)  # Frame stacking
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = np.clip(total_reward, -1, 1)  # Reward clipping
        total_reward = torch.tensor(total_reward).view(1, -1).float().to(self.device)
        
        done = torch.tensor(done).view(1, -1).to(self.device)

        return max_frame, total_reward, done, info
    
    
    def reset(self):
        self.frame_buffer.clear()
        observation, _ = self.env.reset()
        self.lives = self.env.ale.lives()

        self.frame_buffer.append(observation)
        observation = self.process_observation(observation)
        return observation.to(self.device)


    def process_observation(self, observation):
        
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape).convert("L")
        img = np.array(img) / 255.0  # Normalize
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            