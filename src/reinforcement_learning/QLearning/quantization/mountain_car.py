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

env = gym.make("MountainCar-v0")

# CONSTANTS Tweak for better learning
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
DISCRETE_OS_SIZE = [20, 20]

# CONSTANTS Visual
SHOW_EVERY = 5000

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def main():
    for episode in range(EPISODES):
        render = not bool(episode % SHOW_EVERY)
        do_episode(episode, render)

    env.close()


def do_episode(episode_num, render=False):
    global epsilon

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
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            x = q_table[new_discrete_state]
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print(f"EPISODE: {episode_num}: SUCCESSFULLY reached flag!")

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode_num >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


if __name__ == '__main__':
    main()