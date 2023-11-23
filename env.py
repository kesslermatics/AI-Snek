import gym
import numpy as np
import pygame
import sys
import random
from gym import spaces


class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    speed_of_snake = 40

    # Window size
    window_size_x = 480
    window_size_y = 280

    # Colors (R, G, B)
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)

    # Action Constants
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    reward_for_eating = 100
    reward_for_dying = -10
    reward_for_moving_towards_food = 1
    reward_for_moving_away_from_food = -1

    # The number of actions are defined for the agent to know how many actions he can do
    number_of_actions = 4
    # The number of observations are the amount of information the agent has about the current environment
    number_of_observations = 4

    def __init__(self):
        super(SnekEnv, self).__init__()

        # Initialize game
        pygame.init()
        pygame.display.set_caption('AI Snek')

        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((self.window_size_x, self.window_size_y))

        self.reset()

        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.number_of_observations,), dtype=np.float32)

    def reset(self):
        self.counter = 0
        self.spawn_snake()
        self.spawn_food()
        self.score = 0
        self.game_over = False
        self.direction = 'RIGHT'
        self.change_to = self.direction

        return np.array(
            [
                self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]
                ],
                dtype=np.float32)

    def step(self, action):
        self.counter += 1
        reward = 0
        if self.counter > 100:
            reward = self.reward_for_dying
            return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]],
                            dtype=np.float32), reward, True, {}
        if action == self.UP:
            self.change_to = 'UP'
        if action == self.DOWN:
            self.change_to = 'DOWN'
        if action == self.LEFT:
            self.change_to = 'LEFT'
        if action == self.RIGHT:
            self.change_to = 'RIGHT'
        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Moving the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10

        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.counter = 0
            self.score += 1
            self.spawn_food()
        else:
            self.snake_body.pop()

        # Game Over conditions
        # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.window_size_x - 10:
            self.game_over = True
            reward = self.reward_for_dying
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.window_size_y - 10:
            self.game_over = True
            reward = self.reward_for_dying
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.game_over = True
                reward = self.reward_for_dying
        if reward == 0:
            if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
                reward = self.reward_for_eating
            elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) > abs(
                    self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
                reward = self.reward_for_moving_away_from_food
            elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) < abs(
                    self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
                reward = self.reward_for_moving_towards_food

        self.prev_snake_pos = self.snake_pos.copy()
        done = self.game_over
        info = {}
        return np.array(
            [
                    self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]
                    ],
                    dtype=np.float32), reward, done, info

    def render(self, mode='human'):
        # Render visual graphics
        self.game_window.fill(self.white)
        # Render snake
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Render food
        pygame.draw.rect(self.game_window, self.red, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
        # self.show_score(1, self.white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        self.fps_controller.tick(self.speed_of_snake)

    def close(self):
        pygame.quit()
        sys.exit()

    def show_score(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.window_size_x / 2, 15)
        else:
            score_rect.midtop = (self.window_size_x / 2, self.window_size_y / 1.25)
        self.game_window.blit(score_surface, score_rect)
        # pygame.display.flip()

    def spawn_food(self):
        food_in_snake = True
        while food_in_snake:
            food_in_snake = False
            self.food_pos = [
                random.randrange(1, (self.window_size_x // 10)) * 10,
                random.randrange(1, (self.window_size_y // 10)) * 10
            ]
            for i in self.snake_body:
                if i[0] == self.food_pos[0] and i[1] == self.food_pos[1]:
                    food_in_snake = True

    def spawn_snake(self):
        self.snake_pos = [100, 50]
        self.prev_snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]

    def is_snake_near_to_left_wall(self):
        x_pos = self.snake_pos[0]
        return (self.window_size_x - x_pos) < x_pos

    def is_snake_near_to_top_wall(self):
        y_pos = self.snake_pos[1]
        return (self.window_size_y - y_pos) < y_pos

