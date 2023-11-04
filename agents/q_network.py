import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    A simple neural network for approximating Q-values in reinforcement learning.
    This network takes an input representing the state of the environment and outputs
    Q-values for each possible action.
    """
    
    def __init__(self, input_dim, action_dim):
        """
        Initialize the neural network with two hidden layers.
        
        :param input_dim: The dimensionality of the input (the state space size).
                          It defines how many input neurons there are.
        :param action_dim: The dimensionality of the output (the action space size).
                            It defines how many output neurons there are, each representing the
                            Q-value for a particular action.
        """
        super(QNetwork, self).__init__()
        # The first fully connected layer takes the input and passes it through 128 neurons.
        # The number 128 is chosen for the hidden layer size and can be adjusted.
        # Larger sizes may allow the network to learn more complex patterns but may also lead to overfitting.
        self.fc1 = nn.Linear(input_dim, 128)

        # The second fully connected layer reduces the size from 128 to 64 neurons.
        # This layer further processes the information en route to the output.
        self.fc2 = nn.Linear(128, 64)

        # The final fully connected layer reduces the network down to the size of the action dimension.
        # Each neuron in this layer corresponds to the Q-value for a particular action.
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        
        :param x: The input state for which Q-values are to be predicted.
        """
        # The input x is passed through the first layer and then a ReLU activation function.
        # ReLU (Rectified Linear Unit) is used to introduce non-linearities into the model, making it capable
        # of learning more complex functions.
        x = torch.relu(self.fc1(x))

        # The activation output is passed through the second layer and ReLU activation.
        # Again, this introduces non-linearities to the learning process.
        x = torch.relu(self.fc2(x))

        # The output of the second ReLU activation is then passed to the final layer.
        # There is no activation function here because this layer outputs the actual Q-values.
        return self.fc3(x)
