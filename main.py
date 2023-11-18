import sys
from env import SnekEnv
from stable_baselines3 import PPO

env = SnekEnv()

if (sys.argv[1] == "True"):
    model = PPO('MlpPolicy', env, verbose=1, batch_size=256)
    model.learn(total_timesteps=600000)
    model.save("training/snake_ai_model_PPO")

model = PPO.load("training/snake_ai_model_PPO")

obs = env.reset()
while (True):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()