import sys
from env import SnekEnv
from stable_baselines3 import PPO

env = SnekEnv()

model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=100000)
model.save("snake_results")
#model = PPO.learn("snake_results")

obs = env.reset()
done = False

for i in range(int(sys.argv[1])):
    print(f"Episode: {i}")
    while (done == False):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(i)
        if done:
            obs = env.reset()
    done = False