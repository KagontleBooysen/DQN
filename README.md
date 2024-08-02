# Hospital Environment DQN Agent

This project involves training a Deep Q-Network (DQN) agent to navigate a complex hospital environment. The environment is represented as a grid with various elements, including an agent (doctor), medicine cabinet, doctors, nurses, and beds. The agent's goal is to reach the medicine cabinet while avoiding collisions with doctors and nurses.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KagontleBooysen/DQN.git
   cd hospital-dqn
   
# Create a virtual environment:
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`

# Install the required packages:
pip install -r requirements.txt

# Note: If requirements.txt is not provided, you can manually install the necessary packages:
pip install numpy gym keras-rl2

# File Structure
- hospital_env.py: Defines the custom environment for the hospital.
- train.py: Script for training the DQN agent.
- play.py: Script for running the trained agent in the environment.

# Usage
# Training the Agent
To train the agent, run: python train.py

# This script will:

- Create an instance of the HospitalEnv environment.
- Build and compile a neural network model.
- Train the DQN agent using the environment.
- Save the trained model weights to dqn_hospital_weights.h5f.

# Running the Trained Agent
To run the trained agent in the environment, run: python play.py

# This script will:

- Create an instance of the HospitalEnv environment.
- Load the trained model weights from dqn_hospital_weights.h5f.
- Simulate and visualize the agent's performance over a specified number of episodes.
  
# Environment Details
The hospital environment (HospitalEnv) is a grid where the agent can move in four directions (up, down, left, right) or stay in place. The goal is to reach the medicine cabinet while avoiding doctors and nurses. The environment is initialized with random positions for the doctors, nurses, and beds.

# Reward Structure

- Reaching the medicine cabinet: +10
- Colliding with a doctor or nurse: -1 and end of episode
- Regular move: -0.1 per step to encourage quicker completion

# Termination Conditions

- Reaching the medicine cabinet
- Colliding with a doctor or nurse
- Exceeding the maximum number of steps (100)

# Rendering
The environment can be rendered to visualize the agent's movements and the positions of other elements. The following emojis represent the elements:

- Agent: 🙂 (doctor)
- Medicine Cabinet: 💊
- Doctors: 👨‍⚕️
- Nurses: 👩‍⚕️
- Beds: 🛏

# Example Output
After running play.py, you should see an output similar to:




  
