import gymnasium as gym
import random

env = gym.make("SpaceInvaders-v0", render_mode="human")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

print(env.env.get_action_meanings())
print(env.action_space.n)

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info, prob = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
