# Tutorial by www.pylessons.com implementation for pytorch

import random
import gym
import numpy as np
from collections import deque
from torch import nn, optim, zeros, from_numpy, amax, argmax
from torch import load as load_model
from torch import save as save_model
#from sys import exit

def getModel(input_shape, action_space):
    model = nn.Sequential(
          nn.Linear(input_shape, 512, True),
          nn.ReLU(),
          nn.Linear(512, 256, True),
          nn.ReLU(),
          nn.Linear(256, 64, True),
          nn.ReLU(),
          nn.Linear(64, action_space, True),
        )
    return model

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 2000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = getModel(input_shape=self.state_size, action_space = self.action_size)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.criterion = nn.MSELoss()

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        print("Model saved!")
        save_model(self.model, name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return argmax(self.model(state)).item()

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        batch_size = min(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, batch_size)

        state = zeros((batch_size, self.state_size))
        next_state = zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        predicted = self.model(state)

        # do batch prediction to save speed
        target = self.model(state)
        target_next = self.model(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (amax(target_next[i]))

        # Train the Neural Network with batches
        self.optimizer.zero_grad()
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()

    def run(self):
        self.model.train()
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            state = from_numpy(state).float()
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                next_state = from_numpy(next_state).float()
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        return
                self.replay()

    def test(self):
        self.model.eval()
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            state = from_numpy(state).float()
            done = False
            i = 0
            while not done:
                self.env.render()
                action = argmax(self.model(state)).item()
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                state = from_numpy(state).float()
                i += 1
                if done:
                    print(f"episode: {e}/{self.EPISODES}, score: {i}")
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    try:
        agent.run()
        agent.test()
    finally:
        agent.env.close()
