# Importing necessary libraries
import gym
import sys
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from snake_env import SnakeEnv
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Activate interactive mode for matplotlib
plt.ion()

# Create the environment
env = SnakeEnv()

# Instantiate the agent
model = DQN(MlpPolicy, env, verbose=1, buffer_size=300000, learning_rate=0.01, exploration_initial_eps=1, exploration_final_eps=0.05, exploration_fraction=0.5)

# Train the agent
model.learn(total_timesteps=100000)

# Setting up the plot for live-updating during the training process
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title('Training Progress')
ax.set_xlabel('Episode')
ax.set_ylabel('Score')
xs, ys = [], []

# Evaluate the trained agent
episodes = int(sys.argv[2])
for episode in range(episodes):
    print(f"Episode: {episode}")
    obs = env.reset()
    done = False
    episode_rewards = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
    xs.append(episode)
    ys.append(episode_rewards)
    ax.plot(xs, ys, color='blue')
    plt.draw()

# Show the plot at the end of the training
plt.show(block=True)
