import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0")
env.render()

action_size = env.action_space.n
state_size = env.observation_space.n

qTable = np.zeros((state_size, action_size))

total_episodes = 15000
learning_rate = 0.8
max_steps = 99
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

rewards = []

for episode in range(total_episodes):
	state = env.reset()
	step = 0
	done = False
	total_rewards = 0

	for step in range(max_steps):
		exp_exp_tradeoff = random.uniform(0, 1)
		if exp_exp_tradeoff>epsilon:
			action = np.argmax(qTable[state, :])
		else:
			action = env.action_space.sample()

		new_stae, reward, done, info = env.step(action)
		qTable[state, action] = qTable[state, action] + learning_rate*(reward + 
			gamma*np.max(qTable[new_stae, :]) - qTable[state, action])
		total_rewards += reward
		state = new_stae
		if done == True:
			break
	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
	rewards.append(total_rewards)

print("Score over time: "  + str(sum(rewards)/total_episodes))

env.reset()

for episode in range(10):
	state = env.reset()
	step = 0
	done = False

	for step in range(max_steps):
		action = np.argmax(qTable[state, :])
		new_state, reward, done, info = env.step(action)
		if done == True:
			env.render()
			print("No. of steps: ", step)
			break
		state = new_state

env.close()
