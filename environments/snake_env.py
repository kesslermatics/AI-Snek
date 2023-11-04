import numpy as np
import gym
from gym import spaces
from random import randrange
import os
import time

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.snake = [np.array([10, 10])]
        self.aim = np.array([0, 1])
        self.grid_size = 20
        self.food = np.array([randrange(0, self.grid_size), randrange(0, self.grid_size)])
        self.reward = 0
        self.step_count = 0
        self.max_steps = 200
        self.start_time = time.time()
        self.done = False
        self.action_space = spaces.Discrete(3) # Left, Forward, Right
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.grid_size, self.grid_size), dtype=np.int_)

    def reset(self):
        print("Reset")
        self.snake = [np.array([10, 10])]
        self.aim = np.array([0, -1])
        self.done = False
        self.start_time = time.time()
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for segment in self.snake:
            state[segment[0]][segment[1]] = 1
        state[self.food[0]][self.food[1]] = 2
        return state

    def step(self, action, episode, current_reward):
        reward_for_eating = 10
        reward_for_dying = -10
        reward_for_moving = 1

        self.step_count += 1

        # Aktualisierung der Bewegungsrichtung basierend auf der aktuellen Richtung und der ausgewählten Aktion
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

        head = self.snake[-1] + self.aim
        if not self._inside(head) or any((segment == head).all() for segment in self.snake):
            self.done = True
            self.reward = reward_for_dying 
        else:
            if (head == self.food).all():
                self.step_count = 0
                self.reward = reward_for_eating 
                self.snake.append(head)
                while True:
                    self.food = np.array([randrange(0, self.grid_size), randrange(0, self.grid_size)])
                    if not any((segment == self.food).all() for segment in self.snake):
                        break
            else:
                self.snake.append(head)
                self.snake.pop(0)
                self.reward = reward_for_moving
        
        if (self.step_count >= self.max_steps):
            self.done = True
            self.reward = reward_for_dying
            self.step_count = 0
        
        os.system('cls' if os.name == 'nt' else 'clear') 
        print(f"Episode: {episode}")
        print(f"Food: {self.food}")
        print(f"Snake: {self.snake}")
        print(f"Elapsed Time: {str(round(time.time() - self.start_time, 2))}s")
        print(f"Current Reward: {current_reward}")
        grid_display = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for segment in self.snake:
            grid_display[segment[0]][segment[1]] = 'S'
        
        grid_display[self.food[0]][self.food[1]] = 'F'
        
        for row in grid_display:
            print(' '.join(row))

        return self._get_state(), self.reward, self.done, {}

    def _inside(self, head):
        return 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size

    def render(self, mode='human'):
        pass

    def close(self):
        pass
