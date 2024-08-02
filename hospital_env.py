import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class HospitalEnv(gym.Env):
    def __init__(self):
        super(HospitalEnv, self).__init__()
        self.grid_size = 10  # Grid size 10x10
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.grid_size, self.grid_size), dtype=np.int32)
        self.action_space = spaces.Discrete(5)  # 4 directions + stay

        # Initial positions
        self.agent_pos = [0, 0]
        self.medicine_cabinet_pos = [self.grid_size - 1, self.grid_size - 1]

        # Randomly place doctors, nurses, and beds
        self.num_doctors = 5
        self.num_nurses = 5
        self.num_beds = 5
        self.doctor_positions = self._random_positions(self.num_doctors)
        self.nurse_positions = self._random_positions(self.num_nurses)
        self.bed_positions = self._random_positions(self.num_beds)

        self.max_steps = 100  # Maximum steps per episode
        self.steps = 0

    def _random_positions(self, num):
        positions = []
        while len(positions) < num:
            pos = [self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size)]
            if pos != self.agent_pos and pos != self.medicine_cabinet_pos and pos not in positions:
                positions.append(pos)
        return positions

    def reset(self):
        self.agent_pos = [0, 0]  # Reset agent position
        self.steps = 0
        self.doctor_positions = self._random_positions(self.num_doctors)
        self.nurse_positions = self._random_positions(self.num_nurses)
        self.bed_positions = self._random_positions(self.num_beds)
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        obs[tuple(self.agent_pos)] = 1  # Mark agent position
        obs[tuple(self.medicine_cabinet_pos)] = 2  # Mark medicine cabinet
        for pos in self.doctor_positions:
            obs[tuple(pos)] = 3  # Mark doctors
        for pos in self.nurse_positions:
            obs[tuple(pos)] = 4  # Mark nurses
        for pos in self.bed_positions:
            obs[tuple(pos)] = 5  # Mark beds
        return obs

    def step(self, action):
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        # Action 4: Stay (do nothing)

        self.steps += 1
        done = self.agent_pos == self.medicine_cabinet_pos or self.steps >= self.max_steps

        if self.agent_pos == self.medicine_cabinet_pos:
            reward = 10
        elif self.agent_pos in self.doctor_positions or self.agent_pos in self.nurse_positions:
            reward = -1
            done = True
        else:
            reward = -0.1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[tuple(self.agent_pos)] = '🙂'
        grid[tuple(self.medicine_cabinet_pos)] = '💊'
        for pos in self.doctor_positions:
            grid[tuple(pos)] = '👨‍⚕️'
        for pos in self.nurse_positions:
            grid[tuple(pos)] = '👩‍⚕️'
        for pos in self.bed_positions:
            grid[tuple(pos)] = '🛏'
        for row in grid:
            print(' '.join(row))
        print(f"Steps: {self.steps}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
