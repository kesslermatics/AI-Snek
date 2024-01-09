import sys
from env import SnekEnv
from stable_baselines3 import PPO
from training_callback import TrainingCallback

# Instantiate snake game
env = SnekEnv()

# Initialize tensorboard, model and log file paths
CHECKPOINT_DIR = "./train_PPO_final/"
LOG_DIR = "./logs/"
callback = TrainingCallback(check_freq=1000000, save_path=CHECKPOINT_DIR)

model = PPO('MlpPolicy', env=env, verbose=1, batch_size=256, tensorboard_log=LOG_DIR)

if len(sys.argv) == 1:
    model.learn(total_timesteps=100000000, callback=callback)
else:
    model = PPO.load(f"{CHECKPOINT_DIR}/4000000")

obs = env.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
