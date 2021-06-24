from experimantal import ActorNetwork, ValueNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym
from torch.optim.lr_scheduler import StepLR

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


STATE_DIM = 4
ACTION_DIM = 2
STEP = 5000
SAMPLE_NUMS = 10

temperature = 1.0
latent_dim = 30 # 50
categorical_dim = 2
HIDDEN_SIZE= 32 #64


actor_network = ActorNetwork(STATE_DIM,HIDDEN_SIZE,ACTION_DIM)
actor_network.load_state_dict(torch.load("actor_network_best"))
actor_network.eval()

value_network = ValueNetwork(input_size = STATE_DIM,hidden_size = HIDDEN_SIZE,output_size = 1, action_size=ACTION_DIM)
value_network.load_state_dict(torch.load("value_network_best"))
value_network.eval()


task = gym.make("CartPole-v0")
state = task.reset()
done = False
cnt = 0
while True:
    cnt += 1
    task.render()
    logits = actor_network(Variable(torch.Tensor([state])))
    # softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
    one_hot_action = gumbel_softmax(logits, 1.0, hard=True)
    # print(softmax_action.data)
    # action = np.argmax(one_hot_action.data.numpy()[0])
    action = torch.argmax(one_hot_action).item()
    observation, reward, done, _ = task.step(action)
    # Lets see how long it lasts until failing
    # """
    if done:
        task.reset()
        print(f"Game lasted {cnt} moves")
        cnt = 0
    else:
        continue
    # """
print(f"Game lasted {cnt} moves")
