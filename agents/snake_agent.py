import numpy as np
import random
from agents.q_network import QNetwork
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.memory = []
        self.memory_size = memory_size
        
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).flatten()
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).flatten()
            next_state = torch.FloatTensor(next_state).flatten()
            reward = torch.tensor([reward], dtype=torch.float32)
            
            target = self.model(state)
            with torch.no_grad():
                target_val = self.target_model(next_state)
            
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * torch.max(target_val)
            
            output = self.model(state)
            loss = self.criterion(output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
