import numpy as np
import gym
from gym import spaces
from random import randrange
import os
import time

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
        self.grid_size = 20  # Define the size of the grid. The environment will be a 20x20 grid
        self.food = np.array([randrange(0, self.grid_size), randrange(0, self.grid_size)])  # Place food at a random location
        self.reward = 0  # Initialize reward
        self.step_count = 0  # Counter for the number of steps taken in the current episode
        self.max_steps = 150  # The maximum number of steps before the episode terminates
        self.start_time = time.time()  # Record the starting time for timing the episode duration
        self.done = False  # Initialize the done flag which signals the end of an episode
        self.action_space = spaces.Discrete(3)  # Redefine the action space to have three actions: left, forward, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.grid_size, self.grid_size), dtype=np.int_)  # The observation will be the grid

    def reset(self):
        """
        Reset the environment to a new initial state.
        """
        self.snake = [np.array([10, 10])]  # Reset the snake's position
        self.aim = np.array([0, -1])  # Reset the snake's direction to upwards
        self.done = False  # Reset the done flag
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

    def step(self, action, episode, current_reward):
        """
        Take a step in the environment given an action.
        """
        reward_for_eating = 30  # Reward for eating food
        reward_for_dying = -10  # Penalty for dying (hitting the wall or self)
        reward_for_moving = -0.01  # Reward for making a move, encourages the snake to keep moving

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

        # Check for collision with the wall or self
        head = self.snake[-1] + self.aim
        if not self._inside(head) or any((segment == head).all() for segment in self.snake):
            self.done = True
            self.step_count = 0
            self.reward = reward_for_dying
        else:
            # If food is eaten, reset step count, increase reward, and respawn food
            if (head == self.food).all():
                self.step_count = 0
                self.reward = reward_for_eating 
                self.snake.append(head)
                while True:
                    self.food = np.array([randrange(0, self.grid_size), randrange(0, self.grid_size)])
                    if not any((segment == self.food).all() for segment in self.snake):
                        break

            # If no food is eaten, just move the snake forward
            else:
                self.snake.append(head)
                self.snake.pop(0)
                self.reward = reward_for_moving

        # Check if the max number of steps has been reached
        if self.step_count >= self.max_steps:
            self.done = True
            self.reward = reward_for_dying
            self.step_count = 0
        
        # Clear the terminal and print the current state of the game
        # This is useful for visualizing the game when playing in the console.
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Episode: {episode}")
        print(f"Food: {self.food}")
        print(f"Snake: {self.snake}")
        print(f"Elapsed Time: {str(round(time.time() - self.start_time, 2))}s")
        print(f"Step Count: {self.step_count}")
        print(f"Current Reward: {current_reward}")
        grid_display = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for segment in self.snake:
            grid_display[segment[0]][segment[1]] = 'S'
        
        grid_display[self.food[0]][self.food[1]] = 'F'
        
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
        pass

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
