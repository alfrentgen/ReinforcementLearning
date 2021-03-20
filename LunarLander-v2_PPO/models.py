import torch
from torch import nn
from collections import OrderedDict
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_shape, action_space):
        super(Actor, self).__init__()
        input_size = 1
        for dim_size in input_shape:
            input_size *= dim_size
        layers = OrderedDict([
            ('lin0', nn.Linear(input_size, 512, True)),
            ('act0', nn.ReLU()),
            ('lin1', nn.Linear(512, 256, True)),
            ('act1', nn.ReLU()),
            ('lin2', nn.Linear(256, 64, True)),
            ('act2', nn.ReLU()),
            ('lin3', nn.Linear(64, action_space, True)),
            ('act3', nn.ReLU()),
            ('act4', nn.Softmax(dim = 1))
            ])

        self.add_module('actor', nn.Sequential(layers))

    def forward(self, x):
        return self.actor(x)

    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[0], y_true[1], y_true[2]
        #print(advantages, prediction_picks, actions)
        #print(type(y_true[0]), type(y_true[1]), type(y_true[2]))
        LOSS_CLIPPING = 0.2
        #C1 = 1
        C2 = 0.001
        epsilon = 1e-10

        prob = y_pred * actions
        prob = torch.clamp(prob, epsilon, 1.0)
        old_prob = prediction_picks * actions
        old_prob = torch.clamp(old_prob, epsilon, 1.0)

        ratio = prob/(old_prob + epsilon)
        #ratio = torch.exp(torch.log(prob) - torch.log(old_prob))
        p1 = ratio * advantages[:, None]
        p2 = torch.clamp(ratio, min = 1 - LOSS_CLIPPING, max = 1 + LOSS_CLIPPING) * advantages[:, None]

        #minimum = torch.minimum(p1, p2).mean()
        #print(minimum)
        loss_clipped = -torch.minimum(p1, p2).mean()
        #print(p1, p2)
        #print(loss_clipped)
        #loss_vfs = C1 * advantages**2 * actions
        #loss_ent = -C2 * (prob * torch.log(prob + epsilon))
        loss_ent = -(y_pred * torch.log(y_pred + epsilon))
        loss_ent = C2 * loss_ent.mean()
        #print(loss_clipped.item(), loss_ent.item())
        loss = loss_clipped - loss_ent #- loss_vfs
        #loss =  -loss.mean()
        #print(loss)
        return loss

    def load_weights(self):
        imported_weights = np.load("actor_weights.dmp.npy", allow_pickle = True)
        with torch.no_grad():
            seq = self.children()
            for ch in seq:
                idx = 0
                for c in ch:
                    if type(c) == nn.Linear:
                        c.weight = nn.Parameter(torch.from_numpy(imported_weights[idx].transpose()))
                        c.bias = nn.Parameter(torch.from_numpy(imported_weights[idx+1]))
                        idx += 2

class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        input_size = 1
        for dim_size in input_shape:
            input_size *= dim_size

        layers = OrderedDict([
            ('lin0', nn.Linear(input_size, 512, True)),
            ('act0', nn.ReLU()),
            ('lin2', nn.Linear(512, 256, True)),
            ('act2', nn.ReLU()),
            ('lin3', nn.Linear(256, 64, True)),
            ('act3', nn.ReLU()),
            ('lin4', nn.Linear(64, 1, True)),
            ])
        self.add_module('critic', nn.Sequential(layers))

    def forward(self, x):
        return self.critic(x)

    #TODO: check if the calculated value is too big
    def critic_ppo2_loss(self, y_true, y_pred, base_values):
        LOSS_CLIPPING = 0.2
        #print(type(y_pred), type(base_values))
        clipped_value_loss = base_values + torch.clamp(y_pred - base_values, -LOSS_CLIPPING, LOSS_CLIPPING)
        #print(y_true)#, type(clipped_value_loss))
        #print(clipped_value_loss)
        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2

        value_loss = 0.5 * torch.mean(torch.maximum(v_loss1, v_loss2))
        #value_loss = torch.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def load_weights(self):
        imported_weights = np.load("critic_weights.dmp.npy", allow_pickle = True)
        with torch.no_grad():
            seq = self.children()
            for ch in seq:
                idx = 0
                for c in ch:
                    if type(c) == nn.Linear:
                        #print(c.weight.shape, imported_weights[idx].transpose().shape)
                        print(c.bias.shape, imported_weights[idx+1].shape)
                        c.weight = nn.Parameter(torch.from_numpy(imported_weights[idx].transpose()))
                        c.bias = nn.Parameter(torch.from_numpy(imported_weights[idx+1]))
                        idx += 2

def getModels(input_shape, action_space):
    actor = Actor(input_shape, action_space)
    critic = Critic(input_shape)
    print(actor)
    print(critic)
    return actor, critic
