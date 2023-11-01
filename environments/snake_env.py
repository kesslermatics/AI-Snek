import numpy as np
import gym
from gym import spaces
from random import randrange

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.food = np.array([0, 0])
        self.snake = [np.array([10, 0])]
        self.aim = np.array([0, -10])
        self.grid_size = 20
        self.reward = 0
        self.done = False
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.grid_size, self.grid_size), dtype=np.int_)

    def reset(self):
        self.food = np.array([0, 0])
        self.snake = [np.array([10, 0])]
        self.aim = np.array([0, -10])
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for segment in self.snake:
            state[segment[0]][segment[1]] = 1
        state[self.food[0]][self.food[1]] = 2
        return state

    def step(self, action):
        if action == 0:  # Right
            self.aim = np.array([10, 0])
        elif action == 1:  # Left
            self.aim = np.array([-10, 0])
        elif action == 2:  # Up
            self.aim = np.array([0, 10])
        elif action == 3:  # Down
            self.aim = np.array([0, -10])

        head = self.snake[-1] + self.aim
        if not self._inside(head) or any((segment == head).all() for segment in self.snake):
            self.done = True
            self.reward = -10
        else:
            if (head == self.food).all():
                self.reward = 10
                self.snake.append(head)
                self.food = np.array([randrange(0, self.grid_size), randrange(0, self.grid_size)])
            else:
                self.snake.append(head)
                self.snake.pop(0)
                self.reward = 1

        return self._get_state(), self.reward, self.done, {}

    def _inside(self, head):
        return 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size

    def render(self, mode='human'):
        pass

    def close(self):
        pass
