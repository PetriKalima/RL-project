import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import collections
from collections import namedtuple
from wimblepong import Wimblepong
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=512):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        convDim = self.getConvDim()
        self.fc1 = nn.Linear(convDim, hidden) #256
        self.fc2 = nn.Linear(hidden, action_space_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)
            elif type(m) is torch.nn.Conv2d:
                torch.nn.init.xavier_normal(m.weight)    

    def getConvDim(self):
        testinput = torch.zeros(1, 2, 100, 100)
        x = self.conv1(testinput)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        convOut = self.conv3(x)
        F.relu(x)
        return int(np.prod(convOut.size()))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
class Agent(object):
    def __init__(self, env, state_space, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.reset()
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.train_device = "cuda"
        #self.train_device = "cpu"                  
        self.name = "DQN.AI"
        self.policy_net = DQN(state_space, 3).to(self.train_device)
        self.target_net = DQN(state_space, 3).to(self.train_device)
        self.epsilon = 1.0 
        self.n_actions = 3
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        #self.memory = ReplayMemory(1000000)
        self.memory = ReplayMemory(5000)
        self.batch_size = 32 
        self.gamma = 0.99
        self.prev_obs = None  
        self.policy_net.eval() 
        self.state = None 

    def reset(self):
        self.state = self.env.reset()

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def replace_targetpolicy(self):
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        sample = random.random()
        if sample >= self.epsilon:
            with torch.no_grad():
                state = self.preprocess(self.state).to(self.train_device)
                q_values = self.policy_net.forward(state)
                action = torch.argmax(q_values).item()
                return action
        else:
            return random.randrange(self.n_actions)
           
    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = self.preprocess(next_state)
        state = self.preprocess(state)
        self.memory.push(state, action, next_state, reward, done)

    def calculate_loss(self, sample):
        batch = Transition(*zip(*sample))
        states = torch.stack(batch.state).to(self.train_device)
        actions = torch.cat(batch.action).to(self.train_device)
        rewards = torch.cat(batch.reward).to(self.train_device)
        done = torch.ByteTensor(batch.done)
        next_states = torch.cat(batch.next_state).to(self.train_device)
        states = states.squeeze(1)
        state_action_values = self.policy_net(states).gather(1, actions.long()).unsqueeze(-1)
        next_state_values = self.target_net(next_states).max(1)[0]
        next_state_values[done] = 0.0
        next_state_values = next_state_values.detach()
        expected_values = next_state_values * self.gamma + rewards
        return nn.MSELoss()(state_action_values, expected_values)   

    def update_network(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        sample = self.memory.sample(self.batch_size)
        loss = self.calculate_loss(sample)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-0.1, 0.1)
        self.optimizer.step()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "model.mdl")

    def load_model(self, modelfile):
        print("Loading model from file ", modelfile)
        self.policy_net.load_state_dict(torch.load(modelfile, map_location=lambda storage, loc: storage))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.1

    def preprocess(self, observation):
        observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = np.concatenate((self.prev_obs, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        stack_ob = stack_ob.transpose(1, 3)
        self.prev_obs = observation
        return stack_ob    
 

