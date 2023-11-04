from environments.snake_env import SnakeEnv
from agents.snake_agent import DQNAgent
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title('Training Progress')
ax.set_xlabel('Episode')
ax.set_ylabel('Score')
xs, ys = [], []

env = SnakeEnv()
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

output = []

episodes = 200
for e in range(episodes):
    start_time = time.time()
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action, episode=e, current_reward=episode_reward)
        
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
        episode_reward += reward

        if done:
            end_time = time.time()
            output.append(f"Episode: {e+1}/{episodes}, Score: {episode_reward}, Elapsed time: {str(round(end_time - start_time, 2))}s")
            agent.update_target_model()

            xs.append(e+1)
            ys.append(episode_reward)
        
            ax.plot(xs, ys, color='blue')
            plt.draw()
        
            clear_output(wait=True)
            break

for text in output:
    print(text)
plt.show(block=True)
