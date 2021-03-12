import torch
from torch import nn
from collections import OrderedDict

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
        #print(type(y_true[0]), type(y_true[1]), type(y_true[2]))
        LOSS_CLIPPING = 0.2
        #C1 = 1
        C2 = 0.001
        epsilon = 1e-10

        prob = y_pred * actions
        #print(type(prediction_picks))
        old_prob = prediction_picks * actions
        r = prob/(old_prob + epsilon)
        p1 = r * advantages[:, None]
        p2 = torch.clamp(r, min = 1 - LOSS_CLIPPING, max = 1 + LOSS_CLIPPING) * advantages[:, None]

        loss_clipped = torch.minimum(p1, p2)
        #loss_vfs = C1 * advantages**2 * actions
        loss_ent = -C2 * (prob * torch.log(prob + epsilon))
        loss = loss_clipped + loss_ent #- loss_vfs
        loss =  -loss.mean()
        return loss

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

    def critic_ppo2_loss(self, y_true, y_pred, base_values):
        LOSS_CLIPPING = 0.2
        #print(type(y_pred), type(base_values))
        clipped_value_loss = base_values + torch.clamp(y_pred - base_values, -LOSS_CLIPPING, LOSS_CLIPPING)
        #print(type(y_true), type(clipped_value_loss))
        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2

        value_loss = 0.5 * torch.mean(torch.maximum(v_loss1, v_loss2))
        #value_loss = torch.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

def getModels(input_shape, action_space):
    actor = Actor(input_shape, action_space)
    critic = Critic(input_shape)
    print(actor)
    print(critic)
    return actor, critic
