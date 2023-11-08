import torch
import torch.nn as nn

class SimpleQNetwork(nn.Module):
    """
    A simpler neural network for approximating Q-values in reinforcement learning.
    This network will have only one hidden layer.
    """
    
    def __init__(self, input_dim, action_dim):
        """
        Initialize the neural network with one hidden layer.
        
        :param input_dim: The dimensionality of the input (the state space size).
        :param action_dim: The dimensionality of the output (the action space size).
        """
        super(SimpleQNetwork, self).__init__()
        # Only one fully connected layer is used here.
        self.fc1 = nn.Linear(input_dim, 64)  # Reduced from 128 to 64 neurons
        self.fc2 = nn.Linear(64, action_dim)  # Directly to action_dim, removed one layer

    def forward(self, x):
        """
        Defines the computation performed at every call.
        
        :param x: The input state for which Q-values are to be predicted.
        """
        x = torch.relu(self.fc1(x))  # ReLU activation function
        return self.fc2(x)  # The final layer outputs the Q-values directly

# To use this network, you would instantiate it and then use it in your DQNAgent like so:
# model = SimpleQNetwork(state_dim, action_dim)
