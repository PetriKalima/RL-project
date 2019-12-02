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
    def __init__(self, action_space_dim, hidden=12):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        convDim = self.getConvDim()
        self.fc1 = nn.Linear(convDim, 64) #256
        self.fc2 = nn.Linear(64, action_space_dim)

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
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.reset()
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        #self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.train_device = "cuda"
        self.train_device = "cpu"                  
        self.name = "DQN.AI"
        self.policy_net = DQN(3).to(self.train_device) #hua
        #self.target_net = DQN(self.env.observation_space.shape, 3).to(self.train_device)
        self.target_net = DQN(3).to(self.train_device) #hua
        self.epsilon = 1.0  # may need to change
        self.n_actions = 3 #maybe change to env size
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025)
        self.memory = ReplayMemory(100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.prev_obs = None  # hua
        self.policy_net.eval()    #hua
        self.state = None  # hua

    def reset(self):
        self.state = self.env.reset()

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def replace_targetpolicy(self): #hua
        
        self.target_net.load_state_dict(self.policy_net.state_dict()) #hua

    def get_action(self, ob=None): # ob ?
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        sample = random.random()
        if sample < self.epsilon:
            with torch.no_grad():
                #x = self.preprocess(ob).to(self.train_device)
                #print(self.state.shape)
                state = self.preprocess(self.state).to(self.train_device)   #hua "Here"
                #state = torch.from_numpy(np.ascontiguousarray(self.state)).float() #hua
                #state = state.view(1,self.state.shape[2], self.state.shape[0], self.state.shape[1]) #hua
                #print("here, state shape:",state.shape)
                q_values = self.policy_net.forward(state)
                #print(q_values)
                action = torch.argmax(q_values).item()
                #print(action)
                return action
        else:
            return random.randrange(self.n_actions)

    def store_transitionO(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(np.ascontiguousarray(next_state)).float()
        state = torch.from_numpy(np.ascontiguousarray(state)).float()
        self.memory.push(state, action, next_state, reward, done)
        
    def store_transition(self, state, action, next_state, reward, done): #hua !!!
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        #next_state = torch.from_numpy(next_state).float()
        next_state = self.preprocess(next_state)
        #state = torch.from_numpy(state).float()
        state = self.preprocess(state)
        self.memory.push(state, action, next_state, reward, done)

    def calculate_loss(self, sample):  # working here now
        batch = Transition(*zip(*sample))
        states = torch.stack(batch.state)  #h? todevice?
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        #print(rewards)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)# need to change to boolean?
        #non_final_mask = torch.logical_not(torch.cat(batch.done))
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states)
        states = states.view(32, 2, 100, 100)
        #states = states.view(32, 3, 200, 200)
        non_final_next_states = non_final_next_states.view(non_final_next_states.shape[0], 2, 100, 100)
        state_action_values = self.policy_net(states).gather(1, actions)
        #print("asd", state_action_values)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        #print(next_state_values)
        expected_values = next_state_values * self.gamma + rewards
        #print(expected_values)
        #print(rewards, expected_values, state_action_values)
        return nn.MSELoss()(state_action_values, expected_values)
        
    def calculate_lossN(self, sample):  # here now working on
        batch = Transition(*zip(*sample))
        states = torch.stack(batch.state)  #h? todevice?
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)# need to change to boolean?
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states)
        states = states.view(32, 3, 200, 200)
        non_final_next_states = non_final_next_states.view(non_final_next_states.shape[0], 3, 200, 200)
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_values = next_state_values * self.gamma + rewards
        #print(rewards, expected_values, state_action_values)
        return nn.MSELoss()(state_action_values, expected_values)


    def update_network(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        sample = self.memory.sample(self.batch_size)
        loss = self.calculate_loss(sample)
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "modelhua26.mdl")

    def load_model(self, modelfile):
        print("Loading model from file ", modelfile)
        self.policy_net.load_state_dict(torch.load(modelfile, map_location=lambda storage, loc: storage))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.1
        return

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
 

