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

# Hyper Parameters
STATE_DIM = 4
ACTION_DIM = 2
STEP = 5000
SAMPLE_NUMS = 15

temperature = 0.25 #1.0
latent_dim = 25 # 50
categorical_dim = 2
HIDDEN_SIZE= 32 #512#128 # 64


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



class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.normalize(x, dim=0)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out)) #log_softmax
        return out


class ValueNetwork(nn.Module):
    """
    Value Network is now a Q-network which gives critic about state-action pairs
    """
    def __init__(self,input_size,hidden_size,output_size, action_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # action
        self.fc1_action = nn.Linear(action_size,hidden_size) # sample size?
        self.fc2_action = nn.Linear(hidden_size,hidden_size)

        # out
        self.output_layer = nn.Linear(1, 1)

    def forward(self, states, actions):
        #actions = actions.squeeze()
        out_state1 = F.relu(self.fc1(states))
        out_state2 = F.relu(self.fc2(out_state1)) # [22,40]
        out_action1 = F.relu(self.fc1_action(actions))
        out_action2 = F.relu(self.fc2_action(out_action1)) # [22,40]

        #state_action = out_state2 + out_action2
        state_action = torch.add(out_state2, out_action2)
        out = torch.tanh(self.fc3(state_action))
        #out = out.squeeze() # die line ist es nicht
        #out = Variable(out.data, requires_grad=True)
        return out


def roll_out(actor_network,task,sample_nums,value_network,init_state):
    #task.reset()
    states = []
    actions = []
    rewards = []
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        logits = actor_network(Variable(torch.Tensor([state]))) #log softmax from actor
        one_hot_action = gumbel_softmax(logits, temperature, hard=True)
        action = torch.argmax(one_hot_action).item()
        next_state, reward, done, _ = task.step(action)
        actions.append(one_hot_action) # zu liste konvertieren zerstört backpropagation
        rewards.append(reward)
        state = next_state
        if done:
            state = task.reset()
            break
    return states,actions,rewards,final_r,state


def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def create_expected_reward(rewards):
    expected_reward = []
    for i in range(len(rewards)):
        expected_reward.append(1 + sum(rewards[i + 1:]))
    return expected_reward


def plot_grad_flow(named_parameters):
    # took from: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


def plot_loss(collected_loss):
    plt.plot(collected_loss, alpha=0.3, color="b")
    plt.title("Value Loss")
    plt.show()


def plot_loss_actor(collected_loss):
    plt.plot(collected_loss, alpha=0.3, color="b")
    plt.title("Actor Loss")
    plt.show()



def main():
    # init a task generator for data fetching
    task = gym.make("CartPole-v0")
    init_state = task.reset()

    # init value network
    value_network = ValueNetwork(input_size = STATE_DIM, hidden_size = HIDDEN_SIZE, output_size = 1, action_size=ACTION_DIM)
    value_network_optim = torch.optim.Adam(value_network.parameters(),lr=0.0001)
    value_network_optim = torch.optim.SGD(value_network.parameters(), lr = 0.00001) #0.1

    # init actor network
    actor_network = ActorNetwork(STATE_DIM,HIDDEN_SIZE,ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.0001)
    actor_network_optim = torch.optim.SGD(actor_network.parameters(), lr=0.0000001) #0.1 # 0.000001

    steps = []
    task_episodes = []
    test_results = []

    collected_value_network_loss = []
    collected_actor_network_loss = []

    for step in range(STEP):
            states, actions, rewards, final_r, current_state = roll_out(actor_network,task,SAMPLE_NUMS,value_network,init_state)
            init_state = current_state
            actions_var = torch.stack((actions)) #Variable(torch.Tensor(actions).view(-1, ACTION_DIM))
            states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))
            expected_rew = create_expected_reward(rewards)
            rew = torch.FloatTensor(expected_rew)

            # train actor network
            actor_network_optim.zero_grad()
            vs = value_network(states_var, actions_var)#.detach() # detach in order not to backprop through critic
            #vs = vs.detach()
            actor_network_loss = -torch.sum(vs)
            #print(actor_network_loss)
            #actor_network_loss.requires_grad = True
            actor_network_loss.backward(inputs = list(actor_network.parameters()), retain_graph=True)
            #actor_network_optim.step()
            #vs = vs.detach()

            # train value network
            value_network_optim.zero_grad()
            values = value_network(states_var, actions_var)
            criterion = nn.MSELoss()
            #value_network_loss = torch.mean(torch.sum(values - rew)**2)
            value_network_loss = criterion(values, rew) #0.5 * torch.mean(torch.pow(values - rew, 2))
            #print(value_network_loss)
            #torch.autograd.set_detect_anomaly(True)
            value_network_loss.backward(inputs=list(value_network.parameters()))
            torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.0001)
            actor_network_optim.step()
            value_network_optim.step()
            collected_value_network_loss.append(value_network_loss)
            collected_actor_network_loss.append(actor_network_loss)

            #plot_grad_flow(actor_network.named_parameters())
            if step%500 == 0:
                print(step)
                #plot_grad_flow(actor_network.named_parameters())
            #    for n,p in value_network.named_parameters():
            #        print(n, p)


            # Testing
            if (step + 1) % 100== 0: #50
                    result = 0
                    test_task = gym.make("CartPole-v0")
                    for test_epi in range(10):
                        state = test_task.reset()
                        for test_step in range(200):
                            logits = actor_network(Variable(torch.Tensor([state])))
                            #softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
                            one_hot_action = gumbel_softmax(logits, temperature, hard=True)
                            #print(softmax_action.data)
                            #action = np.argmax(one_hot_action.data.numpy()[0])
                            action = torch.argmax(one_hot_action).item()
                            next_state, reward, done, _ = test_task.step(action)
                            result += reward
                            state = next_state
                            if done:
                                #task.reset()
                                break
                    print("step:",step+1,"test result:",result/10.0)
                    steps.append(step+1)
                    test_results.append(result/10)

    print(collected_value_network_loss)
    print(collected_actor_network_loss)
    plot_loss(collected_value_network_loss)
    plot_loss_actor(collected_actor_network_loss)


    task = gym.make("CartPole-v0")
    state = task.reset()
    done = False
    cnt = 0
    avg_step = 0
    i = 0
    while True:
        cnt += 1
        task.render()
        logits = actor_network(Variable(torch.Tensor([state])))
        # softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
        one_hot_action = gumbel_softmax(logits, 1.0, hard=True)
        # print(softmax_action.data)
        #action = np.argmax(one_hot_action.data.numpy()[0])
        action = torch.argmax(one_hot_action).item()
        observation, reward, done, _ = task.step(action)
        # Lets see how long it lasts until failing
        #"""
        if done:
            #print(i)
            print(f"Game lasted {cnt} moves")
            avg_step += cnt
            cnt=0
            task.reset()
            i+=1
        if i==100:
            break

        #"""
    task.close()
    #print(f"Game lasted {cnt} moves")
    print("Average steps: ", avg_step/100)

if __name__ == '__main__':
    main()