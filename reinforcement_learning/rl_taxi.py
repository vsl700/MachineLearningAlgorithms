import gym
import numpy as np
from time import sleep
from IPython.display import clear_output


def RenderFrame(i, frame):
    clear_output()
    print(frame['frame'])
    print(f"Timestep: {i + 1}")
    print(f"State: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")
    sleep(.1)


def RenderFrames(frames):
    for i, frame in enumerate(frames):
        RenderFrame(i, frame)


def SolveWithoutRL():
    env.s = 328

    epochs = 0  # Timesteps
    penalties, reward = 0, 0

    frames = []  # For printing an animation of each timestep

    done = False

    while not done:
        clear_output(wait=True)

        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })

        epochs += 1

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}\n\n".format(penalties))

    RenderFrames(frames)


def EvaluateRLAgent(q_table):
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for e in range(episodes + 1):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        if e >= episodes:
            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_epochs / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            if e >= episodes:
                RenderFrame(e, {
                    'frame': env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                })

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs


def SolveWithRL():
    import random
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.6  # Discount factor (the higher, the more forward reward predictive agent)
    epsilon = 0.1  # The lower, the less exploration rather than exploiting highest reward from Q-table

    # For plotting metrics (unused)
    # all_epochs = []
    # all_penalties = []

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished!\n\n")

    EvaluateRLAgent(q_table)


env = gym.make("Taxi-v3").env

env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

encoded_state = env.encode(3, 1, 2, 0)
print("\n\nState:", encoded_state)

env.s = encoded_state
env.render()

sleep(1)

# SolveWithoutRL()

# print("\n\n\n")

SolveWithRL()
