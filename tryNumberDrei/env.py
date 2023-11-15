import gym
import numpy as np
import pygame
import sys
import random
from gym import spaces

class SnekEnv(gym.Env):
    """
    Custom environment for the Snake game that follows the OpenAI Gym interface.
    """

    number_of_actions = 4    
    number_of_observations = 4
    difficulty = 20
    window_size_x = 360
    window_size_y = 240

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    reward_for_dying = -100
    reward_for_eating = 100
    reward_for_moving_away_from_food = -1
    reward_for_moving_towards_food = 1
    reward_for_staying_at_same_distance = 0

    max_steps = 100

    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SnekEnv, self).__init__()

        pygame.init()
        pygame.display.set_caption("AI-Snek")
        self.game_window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        self.fps_controller = pygame.time.Clock()

        self.reset()
        
        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.number_of_observations, ), dtype=np.float32)
        
    def reset(self):
        self.spawn_snake()
        self.spawn_food()
        self.direction = "RIGHT"
        self.change_to = self.direction
        self.step_count = 0

        self.score = 0
        self.done = False

        return np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32)
    
    def step(self, action):
        self.step_count += 1

        if (self.step_count >= self.max_steps):
            self.done = True
            self.reward = self.reward_for_dying
        else:
            self.handle_movement(action)
        
            self.snake_body.insert(0, list(self.snake_pos))
            if (self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]):
                self.score += 1
                self.reward = self.reward_for_eating
                self.step_count = 0
                self.spawn_food()
            else:
                self.snake_body.pop()

            self.check_death()
            self.handle_reward_for_moving()

        self.observation = np.array([self.snake_pos[0], self.snake_pos[1], self.food_pos[0], self.food_pos[1]], dtype=np.float32)
        self.info = {}
        return self.observation, self.reward, self.done, self.info
    
    def render(self, episode, mode='human', close=False):
        self.draw_game()
        
    def close(self):
        pygame.quit()
        sys.exit()

    def handle_movement(self, action):
        if (action == self.LEFT):
            self.change_to = "LEFT"
        if (action == self.RIGHT):
            self.change_to = "RIGHT"
        if (action == self.UP):
            self.change_to = "UP"
        if (action == self.DOWN):
            self.change_to = "DOWN"
        
        if (self.change_to == "UP" and self.direction != "DOWN"):
            self.direction = "UP"
        if (self.change_to == "DOWN" and self.direction != "UP"):
            self.direction = "DOWN"
        if (self.change_to == "LEFT" and self.direction != "RIGHT"):
            self.direction = "LEFT"
        if (self.change_to == "RIGHT" and self.direction != "LEFT"):
            self.direction = "RIGHT"
        
        if (self.direction == "UP"):
            self.snake_pos[1] -= 10
        if (self.direction == "DOWN"):
            self.snake_pos[1] += 10
        if (self.direction == "LEFT"):
            self.snake_pos[0] -= 10
        if (self.direction == "RIGHT"):
            self.snake_pos[0] += 10

        self.prev_snake_pos = self.snake_pos.copy()
        
    def handle_reward_for_moving(self):
        prev_abs_x = abs(self.prev_snake_pos[0] - self.food_pos[0])
        prev_abs_y = abs(self.prev_snake_pos[1] - self.food_pos[1])

        abs_x = abs(self.snake_pos[0] - self.food_pos[0])
        abs_y = abs(self.snake_pos[1] - self.food_pos[1])

        if (abs_x + abs_y > prev_abs_x + prev_abs_y):
            self.reward = self.reward_for_moving_away_from_food
        else:
            self.reward = self.reward_for_moving_towards_food
        
        if (abs_x + abs_y == prev_abs_x + prev_abs_y):
            self.reward = self.reward_for_staying_at_same_distance

    def check_death(self):
        if (self.snake_pos[0] < 0 or self.snake_pos[0] > self.window_size_x - 10) or (self.snake_pos[1] < 0 or self.snake_pos[1] > self.window_size_y - 10):
            self.done = True
            self.reward = self.reward_for_dying
        
        for block in self.snake_body[1:]:
          if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
              self.done = True
    
    def spawn_food(self):
        self.food_pos = [random.randint(1, self.window_size_x), random.randint(1, self.window_size_y)]
    
    def spawn_snake(self):
        self.snake_pos = [int(self.window_size_x / 2), int(self.window_size_y / 2)]
        self.snake_body = [self.snake_pos.copy()]
        self.prev_snake_pos = self.snake_pos.copy()

    def draw_game(self):
        self.game_window.fill((0, 0, 0))
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, (0, 255, 0), pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, (255, 255, 255), pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        self.show_score((255, 255, 255), "consolas", 20)
        pygame.display.update()
        self.fps_controller.tick(self.difficulty)

    def show_score(self, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render("Score : " + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        
        score_rect.midtop = (self.window_size_x / 2 + 30, 15)
        
        self.game_window.blit(score_surface, score_rect)