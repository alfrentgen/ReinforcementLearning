# Tutorial by www.pylessons.com implementation for pytorch

import os, sys
import random
import gym
import pylab
import numpy as np
from collections import deque
from torch import nn, optim, zeros, from_numpy, amax, argmax, full, mean
from torch import load as load_model
from torch import save as save_model
from collections import OrderedDict
#from sys import exit

class DoubleDDQN(nn.Module):
    def __init__(self, input_shape, action_space, dueling):
        super(DoubleDDQN, self).__init__()
        seq_layers = OrderedDict([
            ('lin0', nn.Linear(input_shape, 512, True)),
            ('relu0', nn.ReLU()),
            ('lin1', nn.Linear(512, 256, True)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(256, 64, True)),
            ('relu2', nn.ReLU())
            ])
        
        self.dueling = dueling
        if dueling:
            self.add_module('sequential', nn.Sequential(seq_layers))
            self.add_module('state_value_layer', nn.Linear(64, 1, True))
            self.add_module('action_advantage_layer', nn.Linear(64, action_space, True))
        else:
            seq_layers['lin3'] = nn.Linear(64, action_space, True)
            self.add_module('sequential', nn.Sequential(seq_layers))
        
    def forward(self, x):
        if self.dueling:
            x = self.sequential(x)
            action_advantage = self.action_advantage_layer(x)
            state_value = self.state_value_layer(x)
            #TODO: check q's dimensionality
            q = state_value + (action_advantage - mean(action_advantage))
            return q
        else:
            return self.sequential(x)

def getModel(input_shape, action_space, dueling):
    model = DoubleDDQN(input_shape, action_space, dueling)
    print(model)
    return model

class DQNAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)  
        # by default, CartPole-v1 has max episode steps = 500
        self.env._max_episode_steps = 4000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # minimum exploration probability
        self.epsilon_decay = 0.999 # exponential decay rate for exploration prob
        self.batch_size = 32 
        self.train_start = 1000

        # defining model parameters
        self.ddqn = True # use doudle deep q network
        self.Soft_Update = False # use soft parameter update
        self.dueling = True # use dealing netowrk

        self.TAU = 0.1 # target network soft update hyperparameter

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []
        
        if self.ddqn:
            print("----------Double DQN--------")
            self.Model_name = os.path.join(self.Save_Path,"Dueling DDQN_"+self.env_name+".h5")
        else:
            print("-------------DQN------------")
            self.Model_name = os.path.join(self.Save_Path,"Dueling DQN_"+self.env_name+".h5")
        
        # create main model
        self.model = getModel(input_shape=self.state_size, action_space = self.action_size, dueling=self.dueling)
        self.target_model = getModel(input_shape=self.state_size, action_space = self.action_size, dueling=self.dueling)
        self.target_model.eval()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.criterion = nn.MSELoss()
        
    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            for name in self.target_model.state_dict():
                shape = self.target_model.state_dict()[name].shape
                zeros(size=shape, out=self.target_model.state_dict()[name])
                self.target_model.state_dict()[name].add_(self.model.state_dict()[name].clone())
                self.target_model.state_dict()[name].detach()
            return
        if self.Soft_Update and self.ddqn:
            for name in self.target_model.state_dict():
                self.target_model.state_dict()[name].multiply_(1-self.TAU)
                self.target_model.state_dict()[name].add_(self.model.state_dict()[name].clone().detach() * self.TAU)
                self.target_model.state_dict()[name].detach_()

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
        #print(f'len(minibatch) = {len(minibatch)}')

        state = zeros((batch_size, self.state_size))
        next_state = zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
            
        # do batch prediction to save speed
        predicted = self.model(state)
        target = self.model(state)
        target_next = self.model(next_state)
        target_val = self.target_model(next_state)
        
        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (amax(target_next[i]))

        # Train the Neural Network with batches
        self.optimizer.zero_grad()
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()
        
    def load(self, name):
        self.model = load_model(name)
        self.target_model = load_model(name)

    def save(self, name):
        #print("Model saved!")
        save_model(self.model, name)
        
    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores) / len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = '_soft'
        try:
            pylab.savefig(dqn+self.env_name+softupdate+".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]

    def run(self):
        self.model.train()
        self.target_model.eval()
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
                    # every step update target model
                    self.update_target_model()
                    
                    # every episode, plot the result
                    average = self.PlotModel(i, e)
                     
                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, self.epsilon, average))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model as cartpole-ddqn.h5")
                        self.save(f"cartpole-dueling-ddqn_ep{e}_torch.h5")
                        self.save(f"cartpole-dueling-ddqn_torch.h5")
                        break
                self.replay()

    def test(self):
        self.model.eval()
        self.target_model.eval()
        self.load("cartpole-dueling-ddqn_torch.h5")
        #self.load("cartpole-ddqn_ep{}_torch.h5".format(episode_number))
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
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    try:
        agent.run()
        print('Now testing...')
        agent.test()
    finally:
        agent.env.close()
