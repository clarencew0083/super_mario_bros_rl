# 🎮 Super Mario Bros with Dueling Deep Q-Network (DDQN)

This project applies deep reinforcement learning to train an autonomous agent to play *Super Mario Bros.* using the Dueling Deep Q-Network (DDQN) architecture. Built on the `gym-super-mario-bros` environment and implemented in PyTorch, the agent learns to navigate game levels from raw pixel input and sparse reward signals.

Key features include:

* 📦 **Environment**: `gym-super-mario-bros` from OpenAI Gym Retro https://github.com/Kautenja/gym-super-mario-bros
* 🧠 **Model**: Dueling DQN with convolutional layers and prioritized experience replay
* 🛠️ **Training Tools**: Frame preprocessing (resizing, grayscaling, stacking), epsilon-greedy exploration, and target network updates
* 🧪 **Testing**: Performance evaluated on both seen (1-1) and unseen (8-1) levels
* ⚠️ **Challenges**: Catastrophic forgetting and sparse reward signals highlight the difficulty of generalization in complex environments

---

## 🚀 Installation

Clone the repository and set up the required environment:

```bash
git clone https://github.com/clarencew0083/super_mario_bros_rl.git
cd super_mario_bros_rl
```

### Set up a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## File Directory ##
* [Model](ddqn.py) contains the Dueling Deep Q Network used for training the Mario agent. 
* [Agent](agent.py) contains code the defines the agent, how it steos in the environment and how it is trained 
* [Replay Buffer](replay_buffer.py) contains the code for the prioritized experience replay buffer
* [Helper Functions](ulility.py) contains helper function to save model chkecpoints when training
* [Wapper](ulility.py) conntains a wrapper used for prerocessing

## 📦 Usage

### Run Training

```bash
python smb_proj.py
```

This will train a DDQN agent on level 1-1 of *Super Mario Bros.* using the Right-Only action space.

Alternative, there is a jupyter notebook named demo.ipynb, if you prefer to run code in that manner.



## 🧠 Training Details

* **Episodes**: 5000
* **Optimizer**: Adam (LR = `5e-5`)
* **Replay Buffer Size**: 150,000 transitions
* **Batch Size**: 64
* **Target Network Update Frequency**: Every 2000 steps
* **Epsilon Decay**: Exponentially from 1.0 to 0.05
* **Discount Factor ($\gamma$)**: 0.99
* **Preprocessing**:

  * Resize frames to 84x84
  * Convert to grayscale
  * Stack 4 frames as input

Training was performed using an NVIDIA A100 GPU over \~23 hours wall time.

---

## 📊 Results

* Catastrophic forgetting was observed, where the agent regressed to dying early after previously achieving high scores.
* The agent showed basic transfer capabilities to level 8-1 but struggled with more complex enemy and platform configurations.
* Videos for the trained Mario agent are in the [videos](videos/) folder

---

## 📈 Future Work

* Incorporate **curriculum learning** to continue learning
* Optimize replay buffer retention to avoid forgetting rare, high-value experiences

---

## 📚 References

References

* Christian Kauten. Super Mario Bros for OpenAI Gym. GitHub, 2018.

* Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning,
2013. 

* “Train a Mario-Playing RL Agent — PyTorch Tutorials 1.8.1+Cu102 Documentation.” Pytorch.org, pytorch.org/tutorials/intermediate/mario_rl_tutorial.html.

* Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas. Dueling network architectures for deep reinforcement learning, 2016.

* Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay, 2015.


---

