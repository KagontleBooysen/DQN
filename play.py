import pygame
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

# Environment parameters
GRID_SIZE = 10
CELL_SIZE = 50
BLOCK_SPACING = 5  # Space between blocks
WIDTH = GRID_SIZE * (CELL_SIZE + BLOCK_SPACING) - BLOCK_SPACING
HEIGHT = GRID_SIZE * (CELL_SIZE + BLOCK_SPACING) - BLOCK_SPACING

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # Agent color
RED = (255, 0, 0)    # Obstacle color (Doctor, Nurse, etc.)
BLUE = (0, 0, 255)   # Medical cabinet color
PURPLE = (128, 0, 128) # Patient color
ORANGE = (255, 165, 0) # Diseased patient color
BLACK = (0, 0, 0)

class HospitalWardEnv:
    def __init__(self):
        self.agent_pos = [0, 0]  # Initial agent position
        self.medical_cabinet = [GRID_SIZE - 1, GRID_SIZE - 1]  # Position of the medical cabinet
        self.patients = [
            (2, 2, "Healthy"),
            (4, 5, "Diseased"),
            (6, 1, "Healthy"),
            (7, 7, "Diseased"),
        ]
        self.obstacles = [
            (1, 1, "Doctor"),
            (1, 3, "Nurse"),
            (3, 1, "Radiologist"),
        ]
        self.dynamic_obstacles = [(3, 3), (5, 5)]  # Dynamic obstacles
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]  # Reset agent position
        return self.get_observation()

    def get_observation(self):
        obs = np.zeros((GRID_SIZE, GRID_SIZE))
        obs[tuple(self.agent_pos)] = 1  # Agent's position
        obs[tuple(self.medical_cabinet)] = 2  # Medical cabinet position
        for obs_pos in self.obstacles:
            obs[obs_pos[0], obs_pos[1]] = -1  # Obstacles
        for patient in self.patients:
            if patient[2] == "Diseased":
                obs[patient[0], patient[1]] = -2  # Diseased patient
            else:
                obs[patient[0], patient[1]] = -3  # Healthy patient
        for dyn_obs in self.dynamic_obstacles:
            obs[dyn_obs[0], dyn_obs[1]] = -4  # Dynamic obstacles
        return obs

    def step(self, action):
        # Define movement based on action
        if action in [0, 1, 2, 3]:  # Up, Down, Left, Right
            new_pos = [self.agent_pos[0] + (action - 1) * (action % 2 * 2 - 1), 
                       self.agent_pos[1] + (action - 2) * ((action + 1) % 2 * 2 - 1)]
        else:  # Perform actions (Diagnose, Prescribe, Administer)
            if action == 4:  # Diagnose
                if self.agent_pos in [[patient[0], patient[1]] for patient in self.patients]:
                    patient = next(p for p in self.patients if (p[0], p[1]) == tuple(self.agent_pos))
                    return self.get_observation(), (1 if patient[2] == "Diseased" else -1), False, {}
            elif action == 5:  # Prescribe
                if self.agent_pos in [[patient[0], patient[1]] for patient in self.patients]:
                    return self.get_observation(), 0.5, False, {}  # Reward for prescribing
            elif action == 6:  # Administer
                return self.get_observation(), 0.5, False, {}  # Reward for administering

            return self.get_observation(), -0.1, False, {}  # Invalid action

        # Check for collisions and update position
        if (0 <= new_pos[0] < GRID_SIZE and 
            0 <= new_pos[1] < GRID_SIZE and 
            not any(new_pos == (obs[0], obs[1]) for obs in self.obstacles) and
            not any(new_pos == (dyn_obs[0], dyn_obs[1]) for dyn_obs in self.dynamic_obstacles)):
            self.agent_pos = new_pos

        done = (self.agent_pos == self.medical_cabinet)
        reward = 0.1 if done else -0.01  # Positive reward for reaching the goal, small penalty otherwise
        return self.get_observation(), reward, done, {}

    def update_dynamic_obstacles(self):
        for i, (x, y) in enumerate(self.dynamic_obstacles):
            new_x = x + random.choice([-1, 0, 1])  # Random movement logic
            new_y = y + random.choice([-1, 0, 1])
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                self.dynamic_obstacles[i] = (new_x, new_y)

    def render(self, screen):
        # Clear the screen
        screen.fill(BLACK)

        # Draw obstacles and labels
        font = pygame.font.Font(None, 20)
        for obs_pos in self.obstacles:
            pygame.draw.rect(screen, RED, 
                             (obs_pos[1] * (CELL_SIZE + BLOCK_SPACING), 
                              obs_pos[0] * (CELL_SIZE + BLOCK_SPACING), 
                              CELL_SIZE, CELL_SIZE))
            text = font.render(obs_pos[2], True, WHITE)
            screen.blit(text, (obs_pos[1] * (CELL_SIZE + BLOCK_SPACING) + 5, 
                               obs_pos[0] * (CELL_SIZE + BLOCK_SPACING) + 5))

        # Draw dynamic obstacles
        for dyn_obs in self.dynamic_obstacles:
            pygame.draw.rect(screen, BLACK, 
                             (dyn_obs[1] * (CELL_SIZE + BLOCK_SPACING), 
                              dyn_obs[0] * (CELL_SIZE + BLOCK_SPACING), 
                              CELL_SIZE, CELL_SIZE))

        # Draw patients
        for patient in self.patients:
            color = PURPLE if patient[2] == "Healthy" else ORANGE  # Healthy vs Diseased
            pygame.draw.rect(screen, color, 
                             (patient[1] * (CELL_SIZE + BLOCK_SPACING), 
                              patient[0] * (CELL_SIZE + BLOCK_SPACING), 
                              CELL_SIZE, CELL_SIZE))
            text = font.render(patient[2], True, WHITE)
            screen.blit(text, (patient[1] * (CELL_SIZE + BLOCK_SPACING) + 5, 
                               patient[0] * (CELL_SIZE + BLOCK_SPACING) + 5))

        # Draw agent and label
        pygame.draw.rect(screen, GREEN, 
                         (self.agent_pos[1] * (CELL_SIZE + BLOCK_SPACING), 
                          self.agent_pos[0] * (CELL_SIZE + BLOCK_SPACING), 
                          CELL_SIZE, CELL_SIZE))
        text = font.render('Agent', True, WHITE)
        screen.blit(text, (self.agent_pos[1] * (CELL_SIZE + BLOCK_SPACING) + 5, 
                           self.agent_pos[0] * (CELL_SIZE + BLOCK_SPACING) + 5))

        # Draw medical cabinet and label
        pygame.draw.rect(screen, BLUE, 
                         (self.medical_cabinet[1] * (CELL_SIZE + BLOCK_SPACING), 
                          self.medical_cabinet[0] * (CELL_SIZE + BLOCK_SPACING), 
                          CELL_SIZE, CELL_SIZE))
        text = font.render('Chemotherapy', True, WHITE)
        screen.blit(text, (self.medical_cabinet[1] * (CELL_SIZE + BLOCK_SPACING) + 5, 
                           self.medical_cabinet[0] * (CELL_SIZE + BLOCK_SPACING) + 5))

        # Update the display
        pygame.display.flip()

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(self.state_size, self.state_size)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns the action with the highest value

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    env = HospitalWardEnv()
    
    # DQN parameters
    agent = DQNAgent(GRID_SIZE, 7)  # 7 actions
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, GRID_SIZE, GRID_SIZE])
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)  # Agent decides action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, GRID_SIZE, GRID_SIZE])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Update dynamic obstacles
            env.update_dynamic_obstacles()

            # Render the environment
            env.render(screen)
            pygame.time.delay(10)  # Delay for visualization
            
            total_reward += reward  # Accumulate rewards

        agent.replay(batch_size)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Print total reward for the episode
        print(f"Episode {e+1}/{episodes} finished with total reward: {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()
