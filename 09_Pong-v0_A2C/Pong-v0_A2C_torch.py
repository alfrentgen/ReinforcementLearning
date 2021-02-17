# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 1.15, Keras 2.2.4

import os
import random
import gym
import pylab
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
from collections import OrderedDict
import cv2

class Actor(nn.Module):
    def __init__(self, input_shape, action_space):
        super(Actor, self).__init__()
        input_size = 1
        for dim_size in input_shape:
            input_size *= dim_size
        actor_layers = OrderedDict([
            ('lin0', nn.Linear(input_size, 512, True)),
            ('act0', nn.ELU()),
            ('lin1', nn.Linear(512, action_space, True)),
            ])

        self.add_module('actor', nn.Sequential(actor_layers))

    def forward(self, x):
        return self.actor(x)
    
class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        input_size = 1
        for dim_size in input_shape:
            input_size *= dim_size

        self.add_module('critic', nn.Linear(input_size, 1, True))

    def forward(self, x):
        return self.critic(x)

def getModel(input_shape, action_space):
    actor = Actor(input_shape, action_space)
    critic = Critic(input_shape)
    print(actor)
    print(critic)
    return actor, critic

class A2CAgent:
    # Policy Gradient Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PG parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.max_average = 10000, -21.0 # specific for pong
        self.lr = 0.001

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        # Instantiate games and plot memory
        self.eps = np.finfo(np.float32).eps.item()
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)

        self.save_path = 'Models'        
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.actor_path = os.path.join(self.save_path, f'{self.env_name}_Actor_{self.lr}_torch.h5')
        self.critic_path = os.path.join(self.save_path, f'{self.env_name}_Critic_{self.lr}_torch.h5')

        # Create ActorCritic network model
        self.actor, self.critic = getModel(input_shape=self.state_size, action_space = self.action_size)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr, alpha=0.9, eps=1e-07)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr, alpha=0.9, eps=1e-07)

    def remember(self, state, action, reward):
        # store episode actions to memory
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        state = state.reshape((state.shape[0], self.REM_STEP*self.ROWS*self.COLS)).float()
        prediction = self.actor(state)
        action = Categorical(torch.nn.functional.softmax(prediction, dim = 1)).sample()
        return action.item()

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        std = np.std(discounted_r)
        #prevent division by 0
        if std == 0:
            std = self.eps
        discounted_r /= std
        return discounted_r

    def replay(self):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # reshape memory to appropriate shape for training
        states = torch.from_numpy(np.vstack(self.states))
        states = states.reshape((states.shape[0], self.REM_STEP*self.ROWS*self.COLS)).float()
        action_probs = self.actor(states)
        values = self.critic(states)
        
        # get episode actions
        actions = torch.tensor(data=self.actions, dtype=torch.long)

        # Compute discounted rewards
        discounted_rewards = torch.from_numpy(self.discount_rewards(self.rewards))
        advantages = discounted_rewards - values
        advantages.detach_()
        
        # training PG network
        score = nn.functional.cross_entropy(action_probs, actions, reduction='none') * advantages
        score = score.sum()
        score.backward()
        self.actor_optimizer.step()
        #print(values.dtype)
        #print(discounted_rewards.dtype)
        critic_loss = nn.functional.mse_loss(values[:, 0], discounted_rewards.float())
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []
        
    def load(self, actor_path = None, critic_path = None):
        if not actor_path:
            actor_path = self.actor_path
        if not critic_path:
            critic_path = self.critic_path
        self.actor = torch.load(actor_path)
        self.critic = torch.load(critic_path)
        
    def save(self, actor_path = None, critic_path = None):
        if not actor_path:
            actor_path = self.actor_path
        if not critic_path:
            critic_path = self.critic_path
        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.save_path+"A2C.png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.model_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def getImage(self, frame):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255
        # converting to RGB (OpenCV way)
        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)

        # inserting new frame to free space
        self.image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        return np.expand_dims(self.image_memory, axis=0)

    def reset(self):
        frame = self.env.reset()
        for i in range(self.REM_STEP):
            state = self.getImage(frame)
        return state

    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.getImage(next_state)
        return next_state, reward, done, info
    
    def run(self):
        #self.load()
        self.actor.train()
        self.critic.train()
        for e in range(self.EPISODES):
            state = self.reset()
            done, score, SAVING = False, 0, ''
            while not done:
                self.env.render()
                # Actor picks an action
                action = self.act(torch.from_numpy(state))
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action)
                # Memorize (state, action, reward) for training
                self.remember(state, action, reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print(f'episode: {e}/{self.EPISODES}, score: {score}, average: {average} {SAVING}')

                    self.replay()
        
        # close environemnt when finish training
        self.env.close()

    #TODO: review this function
    def test(self, model_name):
        #self.load()
        self.actor.eval()
        self.critic.eval()
        for e in range(100):
            state = self.reset()
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.act(torch.from_numpy(state))
                state, reward, done, _ = self.step(action)
                score += reward
                if done:
                    print(f'episode: {e}/{self.EPISODES}, score: {score}')
                    break
        self.env.close()

if __name__ == "__main__":
    #env_name = 'PongNoFrameskip-v4'
    env_name = 'PongDeterministic-v4'
    agent = A2CAgent(env_name)
    agent.run()
    agent.test(agent.model_name)
    #agent.test('Models/Pong-v0_PG_2.5e-05.h5')
