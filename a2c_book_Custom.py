import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp #A


temperature = 0.25
latent_dim = 40
categorical_dim = 2


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
    y_hard = (y_hard - y) #.detach() + y
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




class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        # changed to gumbel softmax
        actor = F.log_softmax(self.actor_lin1(y), dim=0) #F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y)) #y.detach
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) #A
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, _ = run_episode(worker_env, worker_model) #B
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards) #C
        counter.value = counter.value + 1 #D


def run_episode2(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float() #A
    values, logprobs, rewards = [],[],[] #B
    done = False
    j=0
    while (done == False): #C
        j+=1
        policy, value = worker_model(state) #D
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample() #E
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done: #F
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards


def update_params(worker_opt, values, logprobs, rewards, clc=0.01, gamma=0.95):
    # reverse order and call .view(-1) to make sure (give us the most recent action), they are flat
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):  # compute return value and append it
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -torch.sum(values)
    #actor_loss = -1 * logprobs * (Returns - values.detach())  # C Detach the values tensor from graph to prevent propagating through the critic head
    criterion = nn.MSELoss()
    critic_loss = criterion(values, Returns)
    #critic_loss = torch.pow(values - Returns, 2)  # D critic wants to learn to predict the return
    loss = actor_loss.sum() + clc * critic_loss.sum()  # E overall loss
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


def run_episode(worker_env, worker_model, N_steps=50):
    raw_state = np.array(worker_env.env.state)
    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    G=torch.Tensor([0]) #A
    while (j < N_steps and done == False): #B
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)

        one_hot_action = gumbel_softmax(logits, temperature, hard=True)
        action = torch.argmax(one_hot_action).item()

        #action_dist = torch.distributions.Categorical(logits=logits) #Categorical(logits=logits)
        #action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action) # (action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else: #C
            reward = 1.0
            G = value#.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G

if __name__ == '__main__':
    MasterNode = ActorCritic()  # A
    MasterNode.share_memory()  # B
    processes = []  # C
    params = {
        'epochs': 10000,
        'n_workers': 5,
    }
    counter = mp.Value('i', 0)  # D
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))  # E
        p.start()
        processes.append(p)
    for p in processes:  # F
        p.join()
    for p in processes:  # G
        p.terminate()

    #print(counter.value, processes[1].exitcode)  # H

    #"""
    env = gym.make("CartPole-v1")
    env.reset()
    cnt = 0
    #for i in range(10000):
    #while True:
    avg = 0
    i = 0
    while True:
        cnt += 1
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits, value = MasterNode(state)
        one_hot_action = gumbel_softmax(logits, temperature, hard=True)
        action = torch.argmax(one_hot_action).item()
        state2, reward, done, info = env.step(action)
        if done:
            i += 1
            #print("Lost")
            env.reset()
            print(f"Game lasted {cnt} moves")
            avg += cnt
            cnt = 0
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()
        if i == 100:
            break
    print("avg: ", avg/100)

    #"""


"""
#Simulated rewards for 3 steps
r1 = [1,1,-1]
r2 = [1,1,1]
R1,R2 = 0.0,0.0
#No bootstrapping
for i in range(len(r1)-1,0,-1):
    R1 = r1[i] + 0.99*R1
for i in range(len(r2)-1,0,-1):
    R2 = r2[i] + 0.99*R2
print("No bootstrapping")
print(R1,R2)
#With bootstrapping
R1,R2 = 1.0,1.0
for i in range(len(r1)-1,0,-1):
    R1 = r1[i] + 0.99*R1
for i in range(len(r2)-1,0,-1):
    R2 = r2[i] + 0.99*R2
print("With bootstrapping")
print(R1,R2)
"""