import gym
from gym import spaces
import numpy as np
from random import randrange, randint
import sys
import time
import os

class SnakeEnv(gym.Env):
    def check_danger(self, direction):
        head_x, head_y = self.snake[0]
        if direction == 'up':
            return int(head_x == 0 or (head_x - 1, head_y) in self.snake[0])
        elif direction == 'down':
            return int(head_x == self.grid_size - 1 or (head_x + 1, head_y) in self.snake[0])
        elif direction == 'left':
            return int(head_y == 0 or (head_x, head_y - 1) in self.snake[0])
        elif direction == 'right':
            return int(head_y == self.grid_size - 1 or (head_x, head_y + 1) in self.snake[0])

    def get_food_direction(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        x_diff = food_x - head_x
        y_diff = food_y - head_y
        return x_diff, y_diff

    """
    Custom environment for the Snake game that follows the OpenAI Gym interface.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        
        self.grid_size = 10  # The size of the square grid
        self.snake = None    # To keep track of the snake's position
        self.food = None     # To keep track of the food's position
        self.done = False    # To track whether the game is over
        self.start_time = time.time()
        self.step_count = 0  # Counter for the number of steps taken in the current episode
        self.reward = 0
        self.max_steps = 500  # The maximum number of steps before the episode terminates
        self.distance_before_move = 0
        self.distance_after_move = 0
        
        self.action_space = spaces.Discrete(3)
        
        # The observation will be the grid itself
        # The observation space is an (n, n, 1) array where 0 is empty, 1 is the snake, and 2 is the food
        self.observation_space = spaces.Box(low=0, high=2, shape=(106,), dtype=np.float32)
        
        self.reset()
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self.snake = [np.array([self.grid_size // 2, self.grid_size // 2])]
        self.food = np.array([randrange(self.grid_size), randrange(self.grid_size)])
        self.done = False
        self.step_count = 0  # Counter for the number of steps taken in the current episode
        self.start_time = time.time()
        self.reward = 0
        self.aim = np.array([0, -1])  # Reset the snake's direction to upwards
        
        return self._get_observation()
    
    def step(self, action):
        reward_for_eating = 20  # Reward for eating food
        reward_for_dying = -20  # Penalty for dying (hitting the wall or self)
        reward_for_moving_towards_food = 1  # Reward for moving towards the food
        reward_for_moving_away_from_food = -1  # Penalty for moving away from the food

        # Calculate the distance to the food before the move
        direction_vectors = [np.array([0, -1]), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]
        self.aim = direction_vectors[action]

        head = self.snake[-1] + self.aim
        self.distance_before_move = np.linalg.norm(self.snake[0] - self.food)
        self.distance_after_move = np.linalg.norm(self.snake[0] - self.food)

        self.step_count += 1  # Increment step count

        if not self._inside(head) or any((segment == head).all() for segment in self.snake):
            self.done = True
            self.reward = reward_for_dying
        else:
            # If food is eaten, reset step count, increase reward, and respawn food
            if (head == self.food).all():
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
                self.distance_after_move = np.linalg.norm(self.snake[0] - self.food)
                # Adjust the reward based on the snake's movement relative to the food
                if self.distance_after_move < self.distance_before_move:
                    # If the snake has moved closer to the food, reward this action
                    self.reward = reward_for_moving_towards_food
                elif self.distance_after_move > self.distance_before_move:
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
            print(f"Food: {self.food}")
            print(f"Snake: {self.snake}")
            print(f"Elapsed Time: {str(round(time.time() - self.start_time, 2))}s")
            print(f"Current Reward: {self.reward}")
            print(f"Distance Delta: {self.distance_after_move - self.distance_before_move}")

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
            
            os.system('cls' if os.name == 'nt' else 'clear')


        # Get the current state of the environment
        observation = self._get_observation()
        
        return observation, self.reward, self.done, {}
    
    def render(self, mode='human', close=False):
        pass
        
    def close(self):
        # Perform any necessary cleanup
        pass

    def _get_observation(self):
        # Get the base grid observation
        grid = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for s in self.snake:
            grid[s[0], s[1], 0] = 1
        grid[self.food[0], self.food[1], 0] = 2

        # Check for danger in all directions
        danger_straight = self.check_danger('up')
        danger_left = self.check_danger('left')
        danger_right = self.check_danger('right')
        danger_back = self.check_danger('down')

        # Get the direction of the food
        food_direction_x, food_direction_y = self.get_food_direction()

        # Combine the grid observation with danger and food direction
        extended_observation = np.array([danger_straight, danger_left, danger_right, danger_back,
                                         food_direction_x, food_direction_y])

        return np.concatenate((grid.flatten(), extended_observation), axis=0)

    def _inside(self, position):
        """
        Determine whether a position is inside the game grid.
        
        Parameters:
        position (np.array): The position to check, typically the head of the snake.
        
        Returns:
        bool: True if the position is inside the grid, False otherwise.
        """
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size