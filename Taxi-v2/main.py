import numpy as np
import gym
import random
import time

env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
print("Action Size = ", action_size)

state_size = env.observation_space.n
print("State Size = ", state_size)

qTable = np.zeros((state_size, action_size))
print(qTable)

# Hyperparameters
total_episodes = 50000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# Training the environment
for episode in range(total_episodes):
	state = env.reset()
	step = 0
	done = False
	if episode%1000==0:
		print("Training at episode ", episode)
	for step in range(max_steps):
		exp_exp_tradeoff = random.uniform(0,1)

		if exp_exp_tradeoff > epsilon:
			action = np.argmax(qTable[state,:])
		else:
			action = env.action_space.sample()

		new_state, reward, done, info = env.step(action)
		qTable[state, action] = qTable[state, action] + learning_rate*(reward + 
				gamma*np.max(qTable[new_state,:]) - qTable[state, action])
		state = new_state
		if done == True:
			break

	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

# Playing
env.reset()
rewards = []

for episode in range(total_test_episodes):
	state = env.reset()
	step = 0
	done = False
	total_rewards = 0
	print("Playing at episode ", episode)

	for step in range(max_steps):
		env.render()
		action = np.argmax(qTable[state, :])
		new_state, reward, done, info = env.step(action)
		total_rewards += reward
		if done == True:
			rewards.append(total_rewards)
			break
		state = new_state
		# To slow down and to visualize game
		# time.sleep(1)

env.close()
print("Score over time: " + str(sum(rewards)/total_test_episodes))

# Learnt from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb