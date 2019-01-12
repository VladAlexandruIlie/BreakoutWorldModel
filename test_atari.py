import gym
import gym_utils


env = gym.make("Breakout-v0")
obs, reward_sum, done = gym_utils.reset_env(env)
steps = 0
while(True):
    obs, reward, done, info = env.step(env.action_space.sample())
    steps += 1
    reward_sum += reward
    env.render()

    if steps >= 512:
        done = True
    if done:

        print('Episode reward: {:5}'.format(reward_sum), end='. ')
        print('Num steps: {:5}'.format(steps), end='. ')
        print()
        obs, reward_sum, done = gym_utils.reset_env(env)
        steps = 0

