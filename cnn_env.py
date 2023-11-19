import gym
import numpy as np
import pygame
import sys
import random
from gym import spaces

class CnnSnekEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    difficulty = 60
    # Window size
    frame_size_x = 360
    frame_size_y = 240
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
  
    def __init__(self):
        super(CnnSnekEnv, self).__init__()
        pygame.init()
        # Initialise game window
        pygame.display.set_caption('Snake Eater')
        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        # Game variables
        self.snake_pos = [100, 50]
        self.prev_snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]

        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.counter = 0
        self.score = 0
        self.game_over = False

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        number_of_actions = 4
        self.action_space = spaces.Discrete(number_of_actions)
    
        self.image_size = (self.frame_size_y, self.frame_size_x, 1)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.image_size, dtype=np.uint8)


    def step(self, action):
      self.counter += 1
      reward = 0
      
      if self.counter > 100:
          reward = -50
          return self.get_grayscale_image(), -100, True, {}
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
          self.food_spawn = False
      else:
          self.snake_body.pop()

      # Spawning food on the screen
      if not self.food_spawn:
          self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
      self.food_spawn = True
      # Game Over conditions
      # Getting out of bounds
      if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
          self.game_over = True
          reward = -50
      if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
          self.game_over = True
          reward = -50
      # Touching the snake body
      for block in self.snake_body[1:]:
          if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
              self.game_over = True
              reward = -50
      if (reward == 0):
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
          reward = 100
        elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) > abs(self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
          reward = -1
        elif abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1]) < abs(self.prev_snake_pos[0] - self.food_pos[0]) + abs(self.prev_snake_pos[1] - self.food_pos[1]):
          reward = 1

      self.prev_snake_pos = self.snake_pos.copy()
      done = self.game_over
      info = {}
      return self.get_grayscale_image(), reward, done, info
  
    def reset(self):
      # Game variables
      self.snake_pos = [100, 50]
      self.prev_snake_pos = [100, 50]
      self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
      self.counter = 0
      self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10,
                       random.randrange(1, (self.frame_size_y // 10)) * 10]
      self.food_spawn = True

      self.direction = 'RIGHT'
      self.change_to = self.direction

      self.score = 0
      self.game_over = False
      observation = self.get_grayscale_image()
      return observation  
    
    def render(self, mode='human'):
      # GFX
      self.game_window.fill(self.white)
      for pos in self.snake_body:
          # Snake body
          # .draw.rect(play_surface, color, xy-coordinate)
          # xy-coordinate -> .Rect(x, y, size_x, size_y)
          pygame.draw.rect(self.game_window, self.black, pygame.Rect(pos[0], pos[1], 10, 10))

      # Snake food
      pygame.draw.rect(self.game_window, self.green, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
      # self.show_score(1, self.white, 'consolas', 20)
      # Refresh game screen
      pygame.display.update()
      # Refresh rate
      self.fps_controller.tick(self.difficulty)
      
    def close (self):
      pygame.quit()
      sys.exit()

    def show_score(self, choice, color, font, size):
      score_font = pygame.font.SysFont(font, size)
      score_surface = score_font.render('Score : ' + str(self.score), True, color)
      score_rect = score_surface.get_rect()
      if choice == 1:
          score_rect.midtop = (self.frame_size_x / 2, 15)
      else:
          score_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 1.25)
      self.game_window.blit(score_surface, score_rect)
      # pygame.display.flip()

    def get_grayscale_image(self):
        # Zeichnen Sie alle Spielelemente auf das Pygame-Fenster
        self.render()  # Diese Methode sollte alle Spielelemente zeichnen

        # Erfassen Sie das Bild von der Pygame-Oberfläche
        image = pygame.surfarray.array3d(pygame.display.get_surface())

        # Konvertieren Sie das Bild zu Graustufen
        grayscale_image = np.mean(image, axis=2)  # Mitteln über die Farbkanäle
        grayscale_image = grayscale_image.astype(np.uint8)  # Konvertieren zu uint8
        
        grayscale_image = np.transpose(grayscale_image)

        # Stellen Sie sicher, dass das Bild die richtigen Dimensionen hat
        grayscale_image = np.expand_dims(grayscale_image, axis=2)  # Fügen Sie eine Kanaldimension hinzu

        return grayscale_image