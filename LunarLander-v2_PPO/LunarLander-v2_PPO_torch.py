import os
import gym
import pylab
import numpy as np
import copy
import torch
from torch import nn, optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
#from multiprocessing import Pipe
#from environment import Environment
from models import getModels

class PPOAgent:
    # Policy Gradient Main Optimization Algorithm
    def __init__(self, env_name):
        self.env_name = env_name
        env = gym.make(self.env_name)
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape
        env.close()
        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle=False
        self.Training_batch = 1000
        self.replay_count = 0
        self.optimizer = optim.RMSprop#optim.Adam
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Instantiate games and plot memory
        self.eps = np.finfo(np.float32).eps.item()
        self.scores, self.episodes, self.average = [], [], []

        self.save_path = 'Models'
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.actor_path = os.path.join(self.save_path, f'{self.env_name}_Actor_{self.lr}_torch.h5')
        self.critic_path = os.path.join(self.save_path, f'{self.env_name}_Critic_{self.lr}_torch.h5')

        # Create ActorCritic network model
        self.actor, self.critic = getModels(input_shape=self.state_size, action_space = self.action_size)
        self.actor_optimizer = self.optimizer(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = self.optimizer(self.critic.parameters(), lr=self.lr)

    def train(self, flag=True):
        if flag:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

    def eval(self):
        self.train(False)

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
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.title(self.env_name+" PPO training cycle", fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decrease learning rate every saved model
            self.lr *= 0.95
        else:
            SAVING = ""

        return self.average_[-1], SAVING

    def act(self, state):
        prediction = self.actor(torch.from_numpy(state))
        action = Categorical(prediction).sample().to(dtype=torch.long)
        action_onehot = nn.functional.one_hot(action, self.action_size)
        return action.item(), action_onehot.clone().detach().numpy(), prediction[0].clone().detach().numpy()

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        std = np.std(discounted_r)
        #prevent division by 0
        if std == 0:
            std = self.eps
        discounted_r /= std
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = torch.from_numpy(np.vstack(states))
        next_states = torch.from_numpy(np.vstack(next_states))
        actions = torch.from_numpy(np.vstack(actions))
        predictions = torch.from_numpy(np.vstack(predictions))
        critic_base_values = self.critic(states).detach()
        #print(predictions)
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages, targets = self.get_gaes(rewards, dones, values.detach().numpy(), next_values.detach().numpy())
        y_true = (torch.from_numpy(advantages).detach(), predictions, actions)
        actor_loss = None
        critic_loss = None
        critic_history = []
        actor_history = []
        for e in range(self.epochs):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # Get Critic network predictions
            y_pred = self.actor(states)
            actor_loss = self.actor.ppo_loss(y_true, y_pred)
            #print(actor_loss)
            actor_history.append(actor_loss.item())
            actor_loss.backward()
            self.actor_optimizer.step()
            #print(critic_base_values)
            values = self.critic(states)
            critic_loss = self.critic.critic_ppo2_loss(torch.from_numpy(targets), values, critic_base_values)
            critic_history.append(critic_loss.item())
            #print(critic_loss)
            critic_loss.backward()
            self.critic_optimizer.step()
            critic_history.append(critic_loss.item())
        print(actor_history)
        print(critic_history)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(actor_loss.detach().numpy()), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(critic_loss.detach().numpy()), self.replay_count)
        self.replay_count += 1

    def run(self): # train only when episode is finished
        env = gym.make(env_name)
        while True:
            state, done, score, SAVING = env.reset(), False, 0, ''
            state = np.reshape(state, [1, self.state_size[0]]) # shape = [1, 8]
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            self.eval()
            while not done:
                env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = env.step(action)
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward

            self.episode += 1
            average, SAVING = self.PlotModel(score, self.episode)
            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
            self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
            self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)

            self.train()
            #for item in [states, actions, rewards, predictions, dones, next_states]:
                #print(type(item[0]))
            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= self.EPISODES:
                break
        env.close()

    def run_dump(self): # train only when episode is finished
        env = gym.make(env_name)
        self.actor.load_weights()
        self.critic.load_weights()
        while True:
            states = np.load("states.dmp.npy", allow_pickle = True)
            next_states = np.load("next_states.dmp.npy", allow_pickle = True)
            rewards = np.load("rewards.dmp.npy", allow_pickle = True)
            dones = np.load("dones.dmp.npy", allow_pickle = True)
            state, done, score, SAVING = states[0], False, 0, ''
            #state = np.reshape(state, [1, self.state_size[0]]) # shape = [1, 8]
            # Instantiate or reset games memory
            actions, predictions = [], []
            self.eval()
            idx = 0
            while not done:
            #while idx < len(dones):
                #env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = next_states[idx], rewards[idx], dones[idx], None
                # Memorize (state, action, reward) for training
                #states.append(state)
                #next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                #rewards.append(reward)
                #dones.append(done)
                predictions.append(prediction)
                # Update current state
                #state = np.reshape(next_state, [1, self.state_size[0]])
                state = next_state
                score += reward
                idx += 1

            self.episode += 1
            average, SAVING = self.PlotModel(score, self.episode)
            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
            self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
            self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)

            self.train()
            #for item in [states, actions, rewards, predictions, dones, next_states]:
                #print(type(item[0]))
            self.replay(states, actions, rewards, predictions, dones, next_states)
            #if self.episode >= self.EPISODES:
                #break
            break
        env.close()

    def run_batch(self): # train every self.Training_batch episodes
        env = gym.make(env_name)
        state = env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            self.eval()
            for t in range(self.Training_batch):
                #env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = env.step(action)
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)

                    state, done, score, SAVING = env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])

            self.train()
            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= self.EPISODES:
                break
        env.close()

if __name__ == "__main__":
    env_name = 'LunarLander-v2'
    agent = PPOAgent(env_name)
    agent.run_dump()
    #agent.run() # train as PPO, train every epesode
    #agent.run_batch() # train as PPO, train every batch, trains better
    #agent.run_multiprocesses(num_worker = 8)  # train PPO multiprocessed (fastest)
    #agent.test()