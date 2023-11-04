import numpy as np
import random
from agents.q_network import QNetwork
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    """
    A deep Q-learning agent that uses two neural networks (a Q-network and a target Q-network)
    to learn the optimal policy.
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, memory_size=10000):
        """
        Initializes the agent with the given parameters.

        :param state_dim: The size of the state space.
        :param action_dim: The size of the action space.
        :param lr: Learning rate for the optimizer.
        :param gamma: Discount factor for future rewards.
        :param epsilon: Exploration rate. This is the probability of choosing a random action.
        :param epsilon_min: Minimum value that epsilon can decay to.
        :param epsilon_decay: The decay rate for epsilon after each training batch.
        :param batch_size: The number of experiences to sample from memory during training.
        :param memory_size: The maximum size of the memory buffer.
        """
        
        # State and action dimensions are required to build the Q-Networks
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Discount factor for calculating future discounted rewards
        self.gamma = gamma
        
        # Parameters for epsilon-greedy strategy to balance exploration and exploitation
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Batch size for sampling from the memory
        self.batch_size = batch_size

        # Initialize replay memory which stores the transitions
        self.memory = []
        self.memory_size = memory_size

        # Initialize the Q-Network and Target Q-Network
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)

        # Optimizer for training the Q-network
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Loss function for training the Q-network
        self.criterion = nn.MSELoss()

    def act(self, state):
        """
        Selects an action based on the current state using an epsilon-greedy approach.
        
        :param state: The current state of the environment.
        :return: The action to be taken.
        """
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            # With probability epsilon, select a random action
            return random.randrange(self.action_dim)
        
        # Otherwise, predict the Q-values of the current state and select the best action
        state = torch.FloatTensor(state).flatten()
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay memory.

        :param state: The starting state.
        :param action: The action taken.
        :param reward: The reward received from taking the action.
        :param next_state: The resulting state after taking the action.
        :param done: Boolean flag indicating if the episode has ended.
        """
        
        # Add the transition to memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Ensure memory does not exceed specified memory size
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def train(self):
        """
        Trains the Q-Network using random samples from the memory.
        """
        
        # Only train if we have enough samples in our memory for a batch
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the memory
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Prepare the data for PyTorch
            state = torch.FloatTensor(state).flatten()
            next_state = torch.FloatTensor(next_state).flatten()
            reward = torch.tensor([reward], dtype=torch.float32)
            
            # Compute the target Q-value
            target = self.model(state)
            with torch.no_grad():
                target_val = self.target_model(next_state)
            
            # Update the target for the action taken
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * torch.max(target_val)
            
            # Compute the loss between the current Q-values and the target Q-values
            output = self.model(state)
            loss = self.criterion(output, target)
            
            # Perform a gradient descent step to minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update the exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
    
    def update_target_model(self):
        """
        Updates the target network by copying the weights from the trained Q-network.
        """
        self.target_model.load_state_dict(self.model.state_dict())
