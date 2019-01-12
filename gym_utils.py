def reset_env(env):
    obs = env.reset()
    reward_sum = 0
    done = False
    return obs, reward_sum, done