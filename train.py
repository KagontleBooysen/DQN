import gym
from hospital_env import HospitalEnv
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Create the environment
env = HospitalEnv()

# Get the number of actions from the environment
nb_actions = env.action_space.n

# Build a neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save the trained policy network
dqn.save_weights('dqn_hospital_weights.h5f', overwrite=True)
