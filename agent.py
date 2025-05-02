import replay_buffer
from ddqn import DuelingDeepQNetwork
import torch
import numpy as np         
  
class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = replay_buffer.PrioritizedReplayBuffer(capacity=150000)
        self.train_start_threshold = 10000
        self.min_epsilon = 0.05
        self.training_steps = 0
        # online network
        self.Q_eval = DuelingDeepQNetwork(self.n_actions, input_dims, lr)
        # target network
        self.Q_next = DuelingDeepQNetwork(self.n_actions, input_dims, lr)


    def act(self, state):
        # pick randomly (with probability epsilon) from valid actions only
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        # otherwise pick the action with the highest Q-value among valid ones
        state = np.array(state)  / 255.0 # Fast and efficient conversion
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.Q_eval.device)

        with torch.no_grad():
            q_values = self.Q_eval.forward(state)

        return torch.argmax(q_values).item()

    def update_target_network(self):
        self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # function for replay buffer
    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)


    def train(self, batch_size):
        if len(self.memory.buffer) < self.train_start_threshold:
            return
        # Compute beta based on training steps
        beta = min(1.0, 0.4 + self.training_steps * (1.0 - 0.4) / 100_000)
        
        self.Q_eval.optimizer.zero_grad()
        #minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta)

        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states) / 255.0, dtype=torch.float).to(self.Q_eval.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.Q_eval.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.Q_eval.device)
        next_states = torch.tensor(np.array(next_states) / 255.0, dtype=torch.float).to(self.Q_eval.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.Q_eval.device)
        weights = torch.tensor(weights, dtype=torch.float).to(self.Q_eval.device)
        indices = np.arange(batch_size)

        # Compute Q-values for current state
        q_values = self.Q_eval.forward(states)
        q_pred = q_values[indices, actions]

        # Compute Q-values for next state
        with torch.no_grad():
            q_eval_next = self.Q_eval(next_states)
            best_actions = torch.argmax(q_eval_next, dim=1)

            # action evaluation: use target network
            q_next_target = self.Q_next(next_states)
            q_target = rewards + self.gamma * q_next_target[indices, best_actions]
            q_target[dones] = rewards[dones]  # No future reward if done

        td_errors = q_pred - q_target
        loss = (weights * td_errors ** 2).mean()
        loss.backward()
        self.Q_eval.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        
        self.training_steps += 1
