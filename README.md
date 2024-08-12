# Hospital "Lung Cancer" Ward Environment DQN Agent

This project involves developing and training a reinforcement learning (RL) agent designed to navigate a simulated hospital "Lung Cancer" ward environment. The agent's primary objectives is to diagnose patients, prescribe treatment, administer medicine, and refer patients to specialists (doctors). If a patient is diagnosed as "unhealthy," the agent is responsible for transporting the patient to the chemotherapy ward for further treatment. The environment is created using Python and Pygame, with the agent trained using a Deep Q-Network (DQN) in TensorFlow. The agent's success is measured by its ability to maximize rewards, which are earned by efficiently diagnosing and treating patients while avoiding unnecessary random movements.  Access the simulation presentation here : https://drive.google.com/file/d/1rCTOGvLQpE2WNc-kqkAdI5PmU_Ugb4FY/view?usp=sharing

# Environment

The hospital "Lung cancer" ward is represented as a grid environment where each cell can contain different entities such as patients, medical professionals, obstacles, and special areas (e.g., the chemotherapy ward). The agent starts at a predefined position and must navigate the grid to interact with patients and provide appropriate care.

# Key Entities

- **Agent(Green)** : Represents the healthcare provider (Machine Learning) responsible for prescribing, reference to a doctor, and administering medicine. If a client is diagnosed as “unhealthy”, the agent takes the patient to the chemotherapy ward.
- **Patients (Purple/Orange)**: Represent healthy or diseased patients that the agent interacts with.
- **Healthy Patients (Purple)**: Patients without any medical conditions.
- **Diseased Patients (Orange**): Patients diagnosed with a health condition requiring further actions.
- **Chemotherapy Ward (Blue)**: A location the agent needs to reach to fetch medical supplies or drugs and take patients for lung cancer treatment.
- **Obstacles (Red)**: Represent doctors, nurses, and other healthcare staff the agent must avoid (or interact with) depending on the scenario.
- **Dynamic Obstacles (Black)**: Moving obstacles that add complexity to the agent's navigation.

  # Agent Actions
The agent can perform the following actions:

1. **Move Up/Down/Left/Right**: Navigate the grid to reach patients or other important locations.
2. **Diagnose**: The agent checks a patient's health status. If the patient is diagnosed as "Diseased," the agent takes further action.
3. **Prescribe**: The agent prescribes treatment to a diseased patient.
4. **Reference to a Doctor**: If the situation requires it, the agent can refer the patient to a doctor for specialized care.
5. **Administer Medicine**: The agent administers the prescribed treatment to the patient.
6. **Transport to Chemotherapy Ward**: If a patient is diagnosed as "Unhealthy" (e.g., having cancer), the agent guides the patient to the chemotherapy ward for treatment.

# Simulation Flow

1. **Diagnosis**: The agent first approaches a patient and performs a diagnosis.

- If the patient is healthy, the agent moves on to the next task.
- If the patient is diseased, the agent proceeds with further actions.
  
2. **Treatment Plan**:

- The agent may prescribe medication or refer the patient to a doctor based on the diagnosis.
- If medication is prescribed, the agent may fetch it from the medical cabinet and administer it to the patient.
- 
3. **Transport to Chemotherapy Ward**:

- If the diagnosis reveals a serious condition (e.g., cancer), the agent transports the patient to the chemotherapy ward for specialized treatment.
  
4. **Goal**: The agent earns rewards for successfully diagnosing, treating, and correctly managing patients. The primary goal is to optimize these rewards by optimizing the actions and paths taken to care for patients.

 # Training and Performance
 
- The agent is trained using Deep Q-Learning, a reinforcement learning technique where the agent learns to take actions based on a Q-value function.
- The environment is updated dynamically with moving obstacles, making the agent's navigation and decision-making process more challenging.
- Over time, the agent reduces random exploration and learns to make optimal decisions with higher confidence.  

## Installation and Running

- Python 3.7+
- Pygame
- TensorFlow
- NumPy
- gym

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
  

# Terminal Conditions

- Taking patients to chemotherapy ward
- Referring patients to doctor
- Administer medicine
- Prescribe medicines to diagnosed patients

# Rendering
The environment can be rendered to visualize the agent's movements and the positions of other elements. The following emojis represent the elements:

- Agent: 🙂 
- Chemotherapy Ward: 💊
- Doctors: 👨‍⚕️
- Nurses: 👩‍⚕️
- Radiologist: 🛏

# Example Output
Training the model before play.py

![Training_model](https://github.com/user-attachments/assets/4d80596b-6983-465b-be2d-7dc74370da8b)

After running play.py, you should see an output similar to:


![Chemo 1](https://github.com/user-attachments/assets/de5ad22e-d16a-4958-b30e-165e0782c77c)






  
