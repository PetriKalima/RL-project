import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from wimblepong import Wimblepong
import random

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.conv1 = nn.Conv2d(state_space_dim[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        convDim = self.getConvDim(state_space_dim)
        self.fc1 = nn.Linear(convDim, 400)
        self.fc2 = nn.Linear(400, action_space_dim)

    def getConvDim(self, shape):
        testinput = torch.zeros(1, shape[2], shape[0], shape[1])
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

class Agent(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.reset()
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        #self.train_device = "cuda"
        self.train_device = "cpu"                  
        self.name = "DQN.AI"
        self.DQN = DQN(self.env.observation_space.shape, 3).to(self.train_device)
        self.epsilon = 0.05
        self.n_actions = 3 #maybe change to env size

    def reset(self):
        self.state = self.env.reset()

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                #x = self.preprocess(ob).to(self.train_device)
                #print(self.state.shape)
                state = torch.from_numpy(self.state).float()
                state = state.view(1,self.state.shape[2], self.state.shape[0], self.state.shape[1])
                #print(state.shape)
                q_values = self.DQN.forward(state)
                print(q_values)
                action = torch.argmax(q_values).item()
                return action
        else:
            return random.randrange(self.n_actions)


    def load_model(self, modelfile):
        print("Loading model from file", modelfile)
        state_dict = torch.load(modelfile)
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
 

