"""
Sentdex yt tutorial: https://www.youtube.com/watch?v=yMk_XtIEzH8&t=909s
Sentdex text tutorial: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/

**Rewards:**
each step: -1
flag: 0

**Formula:**
The DISCOUNT is a measure of how much we want to care about FUTURE reward rather than immediate reward
"""
import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("MountainCar-v0")

# CONSTANTS Visual
EPISODES = 10000
SHOW_EVERY = 5000
STATS_EVERY = 100
RENDER = False
SAVE_QTABLES = True

# STATS
ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "max": [], "min": []}


# CONSTANTS Tweak for better learning
LEARNING_RATE = 0.1
DISCOUNT = 0.95
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def main():
    for episode in range(EPISODES):
        render = not bool(episode % SHOW_EVERY)
        do_episode(episode, render)
    env.close()
    visualize_data()


def do_episode(episode_num, render=False):
    global epsilon

    episode_reward = 0
    if render:
        print(f"EPISODE: {episode_num}")
    init_state = env.reset()
    discrete_state = get_discrete_state(init_state)
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render and RENDER:
            env.render()

        if not done:
            x = q_table[new_discrete_state]
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            # print(f"EPISODE: {episode_num}: SUCCESSFULLY reached flag!")

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode_num >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    add_stats(episode_num, episode_reward)
    save_qtable(q_table, episode_num)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


def add_stats(episode_num, ep_reward):
    ep_rewards.append(ep_reward)
    if not episode_num % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards["ep"].append(episode_num)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["max"].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards["min"].append(min(ep_rewards[-STATS_EVERY:]))
        # print(f"Episode: {episode_num:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}")


def visualize_data():
    plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="average rewards")
    plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max rewards")
    plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min rewards")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def save_qtable(q_table, episode_num):
    if episode_num % 100 == 0:
        folder_path = "qtables/mountain_car/"
        file_path = folder_path + f"{episode_num}-qtable.npy"
        np.save(file_path, q_table)


if __name__ == '__main__':
    main()