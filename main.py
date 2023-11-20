import gymnasium as gym
import random
from tf.keras.optimizers import Adam
from model import build_model
from agent import build_agent

env = gym.make("SpaceInvaders-v0", render_mode="human")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

print(env.get_action_meanings())
print(env.action_space.n)

model = build_model(height, width, channels, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=0.001), metrics=['mae']) 

dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

'''
RANDOM SCENARIO

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
''' 
