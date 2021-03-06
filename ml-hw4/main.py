import gym


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    # env.reset()
    # for _ in range(10000):
    #     env.render()
    #     env.step(env.action_space.sample())  # take a random action
    # env.close()

    env = gym.make('CartPole-v0')
    for i_episode in range(2000):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
