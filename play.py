import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from hospital_env import HospitalEnv

# Create the environment
env = HospitalEnv()

# Get the number of actions from the environment
nb_actions = env.action_space.n

# Build a neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))  # Output layer with number of actions

# Configure the agent with EpsGreedyQPolicy
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()  # Using EpsGreedyQPolicy for exploration
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Load the trained policy network weights (if available)
try:
    dqn.load_weights('dqn_hospital_weights.h5f')
except Exception as e:
    print(f"Could not load weights: {e}")

# Simulate and visualize the agent's performance
num_episodes = 5  # Number of episodes for testing
for episode in range(num_episodes):
    obs = env.reset()  # Reset the environment for each episode
    done = False
    total_reward = 0
    steps = 0

    while not done:
        env.render()  # Render the environment
        print(f"Episode {episode + 1}, Steps: {steps}, Total Reward: {total_reward:.2f}")
        action = input("Enter action (0=Up, 1=Down, 2=Left, 3=Right, 4=Stay): ")

        try:
            action = int(action)
            if action not in [0, 1, 2, 3, 4]:
                print("Invalid action! Please enter a number between 0 and 4.")
                continue
        except ValueError:
            print("Invalid input! Please enter a number between 0 and 4.")
            continue

        obs, reward, done, _ = env.step(action)  # Take a step in the environment
        total_reward += reward
        steps += 1
        print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    print(f"Episode {episode + 1}: Total reward: {total_reward:.2f}, Steps taken: {steps}")

# Close the environment after simulation
env.close()
