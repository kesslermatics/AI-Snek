# Importing necessary classes and functions from other files and libraries.
from environments.snake_env import SnakeEnv
from agents.snake_agent import QLearningAgent  # Make sure to import the correct class
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import sys

# Instantiate the Snake environment.
env = SnakeEnv()
# Calculate the state dimension by multiplying the size of the grid.
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
# Retrieve the number of possible actions from the environment.
action_dim = env.action_space.n
# Create an instance of the QLearning agent with the state and action dimensions.
agent = QLearningAgent(state_dim, action_dim)

fig, ax = plt.subplots(figsize=(10, 5))

# List to store output information for each episode.
output = []
# These lists will store the episodes and corresponding scores for plotting.
xs, ys = [], []

# Set the number of episodes for which the agent will be trained.
episodes = int(sys.argv[2])

for e in range(episodes):
    print(f"Episode: {e}")
    # Record the start time of the episode.
    start_time = time.time()
    # Reset the environment to start a new episode and get the initial state.
    state = env.reset()
    # Initialize the reward accumulated in the episode.
    episode_reward = 0

    # This loop will run until the episode is done.
    while True:
        # The agent chooses an action based on the current state.
        action = agent.act(state)
        # The environment responds to the action with the next state and reward.
        next_state, reward, done, _ = env.step(action, episode=e, current_reward=episode_reward, current_epsilon=agent.epsilon)
        
        # The agent stores the experience in memory.
        agent.remember(state, action, reward, next_state, done)
        # The agent performs a learning step.
        agent.train()
        
        # Update the state for the next iteration.
        state = next_state
        # Accumulate the reward.
        episode_reward += reward

        if done:
            # Update epsilon after each episode.
            agent.update_epsilon()

            end_time = time.time()
            # Append the result of the episode to the output list.
            output.append(f"Episode: {e+1}/{episodes}, Score: {episode_reward}, Elapsed time: {str(round(end_time - start_time, 2))}s")

            # Append the episode number and reward to lists for plotting.
            xs.append(e+1)
            ys.append(episode_reward)

            ax.plot(xs, ys, color='blue')
            plt.draw()
            
            # Clear the previous output to make space for the new one.
            clear_output(wait=True)
            break

# Print out the results of each episode.
for text in output:
    print(text)

print(f"Ending epsilon: {agent.epsilon}")

# Setting up the plot for live-updating during the training process.
ax.set_title('Training Progress')
ax.set_xlabel('Episode')
ax.set_ylabel('Score')

# Display the plot with a blocking call to ensure it stays open at the end of the script.
plt.show(block=True)
