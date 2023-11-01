from environments.snake_env import SnakeEnv
from agents.snake_agent import DQNAgent

env = SnakeEnv()
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

episodes = 50
for e in range(episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
        episode_reward += reward

        if done:
            print(f"Episode: {e+1}/{episodes}, Score: {episode_reward}")
            agent.update_target_model()
            break
