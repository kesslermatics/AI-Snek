import numpy as np
import random
from agents.simple_q_network import SimpleQNetwork
import torch
import torch.nn as nn
import torch.optim as optim

class QLearningAgent:
    """
    A Q-learning agent that uses a single neural network to approximate Q-values.
    """
    
    def __init__(self, state_dim, action_dim, lr=0.7, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.993, batch_size=64, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size
        self.model = SimpleQNetwork(state_dim, action_dim)
        
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).flatten()
        if torch.cuda.is_available():
            state = state.cuda()
        
        q_values = self.model(state)
        print(f"Q-Values: {q_values}")
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

            if torch.cuda.is_available():
                state = state.cuda()
                next_state = next_state.cuda()

            reward = torch.tensor([reward], dtype=torch.float32)
            target = self.model(state)
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state)
                target[action] = reward + self.gamma * torch.max(next_q_values)
            
            output = self.model(state)
            loss = self.criterion(output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
