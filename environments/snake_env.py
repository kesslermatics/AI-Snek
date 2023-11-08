import numpy as np
import gym
from gym import spaces
from random import randint
import os
import time
import sys

class SnakeEnv(gym.Env):
    """
    Custom environment for a Snake game that follows the gym interface.
    This is a simple version of the classic Snake game, where the snake moves in a grid and tries to eat food.
    """

    def __init__(self):
        """
        Initialization of the environment. Sets up the snake, the food, and other environment parameters.
        """
        super(SnakeEnv, self).__init__()
        self.snake = [np.array([10, 10])]  # Initialize the snake at the center of the grid
        self.aim = np.array([0, 1])  # The initial direction of the snake is upwards
        self.grid_size = 10  # Define the size of the grid. The environment will be a 20x20 grid
        self.food = np.array([randint(0, self.grid_size-1), randint(0, self.grid_size-1)])  # Place food at a random location
        self.reward = 0  # Initialize reward
        self.step_count = 0  # Counter for the number of steps taken in the current episode
        self.max_steps = 1000  # The maximum number of steps before the episode terminates
        self.start_time = time.time()  # Record the starting time for timing the episode duration
        self.done = False  # Initialize the done flag which signals the end of an episode
        self.action_space = spaces.Discrete(3)  # Redefine the action space to have three actions: left, forward, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.grid_size, self.grid_size), dtype=np.int_)  # The observation will be the grid

    def reset(self):
        """
        Reset the environment to a new initial state.
        """
        self.snake = [np.array([5, 5])]  # Reset the snake's position
        self.aim = np.array([0, -1])  # Reset the snake's direction to upwards
        self.done = False  # Reset the done flag
        self.reward = 0
        self.start_time = time.time()  # Reset the start time
        return self._get_state()  # Return the initial state

    def _get_state(self):
        """
        Get the current state of the environment, which includes the positions of the snake and food.
        """
        state = np.zeros((self.grid_size, self.grid_size))  # Initialize a grid of zeros
        for segment in self.snake:
            state[segment[0]][segment[1]] = 1  # Mark the snake's position with 1s
        state[self.food[0]][self.food[1]] = 2  # Mark the food's position with a 2
        return state

    def step(self, action, episode, current_reward, current_epsilon):
        """
        Take a step in the environment given an action.
    
        Parameters:
        action (int): The action to be taken by the agent.
        episode (int): The current episode number.
        current_reward (float): The current accumulated reward in the episode.
    
        Returns:
        tuple: A tuple containing the new state, the reward received by taking the action,
           a boolean indicating if the episode has ended, and an empty dictionary.
        """
        reward_for_eating = 10  # Reward for eating food
        reward_for_dying = -10  # Penalty for dying (hitting the wall or self)
        reward_for_moving_towards_food = 1  # Reward for moving towards the food
        reward_for_moving_away_from_food = -1  # Penalty for moving away from the food

        # Calculate the distance to the food before the move
        head = self.snake[-1] + self.aim
        distance_before_move = np.linalg.norm(self.snake[0] - self.food)
        distance_after_move = np.linalg.norm(self.snake[0] - self.food)

        self.step_count += 1  # Increment step count

        # This block updates the direction of movement based on the action and current direction
        # If the snake is moving up and the action is to move left, change direction to left, etc.
        # The logic ensures the snake cannot move in the reverse direction directly.
        if np.array_equal(self.aim, [0, 1]):  # moving up
            if action == 0:  # turn left
                self.aim = np.array([-1, 0])
            elif action == 2:  # turn right
                self.aim = np.array([1, 0])
        elif np.array_equal(self.aim, [0, -1]):  # moving down
            if action == 0:  # turn left
                self.aim = np.array([1, 0])
            elif action == 2:  # turn right
                self.aim = np.array([-1, 0])
        elif np.array_equal(self.aim, [-1, 0]):  # moving left
            if action == 0:  # turn left
                self.aim = np.array([0, -1])
            elif action == 2:  # turn right
                self.aim = np.array([0, 1])
        elif np.array_equal(self.aim, [1, 0]):  # moving right
            if action == 0:  # turn left
                self.aim = np.array([0, 1])
            elif action == 2:  # turn right
                self.aim = np.array([0, -1])

        if not self._inside(head) or any((segment == head).all() for segment in self.snake):
            self.done = True
            self.step_count = 0
            self.reward = reward_for_dying
        else:
            # If food is eaten, reset step count, increase reward, and respawn food
            if (head == self.food).all():
                self.step_count = 0
                self.snake.append(head)
                self.reward = reward_for_eating     
                while True:
                    self.food = np.array([randint(0, self.grid_size-1), randint(0, self.grid_size-1)])
                    if not any((segment == self.food).all() for segment in self.snake):
                        break

            # If no food is eaten, just move the snake forward
            else:
                self.snake.append(head)
                self.snake.pop(0)
                distance_after_move = np.linalg.norm(self.snake[0] - self.food)
                # Adjust the reward based on the snake's movement relative to the food
                if distance_after_move < distance_before_move:
                    # If the snake has moved closer to the food, reward this action
                    self.reward = reward_for_moving_towards_food
                elif distance_after_move > distance_before_move:
                    # If the snake has moved away from the food, penalize this action
                    self.reward = reward_for_moving_away_from_food

        # Check if the max number of steps has been reached
        if self.step_count >= self.max_steps:
            self.done = True
            self.step_count = 0
            self.reward = reward_for_dying
        
        if (sys.argv[1] == "True"):
            # Clear the terminal and print the current state of the game
            # This is useful for visualizing the game when playing in the console.
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Episode: {episode}")
            print(f"Food: {self.food}")
            print(f"Snake: {self.snake}")
            print(f"Elapsed Time: {str(round(time.time() - self.start_time, 2))}s")
            print(f"Step Count: {self.step_count}")
            print(f"Current Reward: {current_reward}")
            print(f"Current Epsilon: {current_epsilon}")
            print(f"Distance Delta: {distance_after_move - distance_before_move}")

        # Initialize the display grid with a border
        grid_display = [[' ' for _ in range(self.grid_size + 2)] for _ in range(self.grid_size + 2)]

        # Set the horizontal border
        for i in range(self.grid_size + 2):
            grid_display[0][i] = grid_display[-1][i] = '-'

        # Set the vertical border
        for i in range(1, self.grid_size + 1):
            grid_display[i][0] = grid_display[i][-1] = '|'

        # Place the snake and food symbols
        for segment in self.snake:
            grid_display[segment[0] + 1][segment[1] + 1] = 'S'  # Adjust for border offset

        grid_display[self.food[0] + 1][self.food[1] + 1] = 'F'  # Adjust for border offset

        # Print the display grid to the console
        for row in grid_display:
            print(' '.join(row))

        return self._get_state(), self.reward, self.done, {}

    def _inside(self, head):
        """
        Check if a position is inside the bounds of the grid.
        """
        return 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size

    def render(self, mode='human'):
        """
        Render the environment to the screen. Currently, this is not implemented.
        """
        

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
