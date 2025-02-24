import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from gym.spaces import MultiDiscrete


class QNetwork(nn.Module):
    """Neural network to approximate the Q-function."""
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Ensure action is an integer (flattened index)
        if isinstance(action, (list, np.ndarray)):
            action = self._flatten_action(action)  # Convert to flattened index
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def _flatten_action(self, action):
        """Convert a unified action (list of length 4) to a flattened index."""
        source_type, source_idx, color, target_row = action
        return (
            (source_type * 5 * 5 * 6) +  # source_type (0 or 1)
            (source_idx * 5 * 6) +        # source_idx (0-4)
            (color * 6) +                 # color (0-4)
            target_row                    # target_row (0-5)
        )


class DQNAgent:
    """DQN agent implementation."""
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=1e-3, batch_size=64, buffer_capacity=10000):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Initialize Q-network and target network
        self.input_size = env.observation_space.shape[0]
        self.output_size = 300  # Total number of actions (2 * 5 * 5 * 6)
        self.q_network = QNetwork(self.input_size, self.output_size)
        self.target_network = QNetwork(self.input_size, self.output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)


    def get_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the best action based on Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update_target_network(self):
        """Update the target network with the Q-network's weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        """Train the Q-network using a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            print("Replay buffer not full yet. Skipping training.")
            return

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))  # Shape: [batch_size, observation_size]
        actions = torch.LongTensor(np.array(actions))  # Shape: [batch_size]
        rewards = torch.FloatTensor(rewards)  # Shape: [batch_size]
        next_states = torch.FloatTensor(np.array(next_states))  # Shape: [batch_size, observation_size]
        dones = torch.FloatTensor(dones)  # Shape: [batch_size]

        # Ensure actions are within valid range
        if (actions >= self.output_size).any():
            invalid_action = actions.max().item()
            raise ValueError(f"Invalid action index: {invalid_action} (max allowed: {self.output_size - 1})")

        # Reshape actions to [batch_size, 1] for gather
        actions = actions.unsqueeze(1)  # Shape: [batch_size, 1]

        # Compute Q-values for current states
        current_q_values = self.q_network(states).gather(1, actions)  # Shape: [batch_size, 1]

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]  # Shape: [batch_size]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # Shape: [batch_size]

        # Compute loss and update the Q-network
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)  # Squeeze to match shapes
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Verbose statements
        print(f"Training Step:")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Epsilon: {self.epsilon:.4f}")
        print(f"  - Avg Reward in Batch: {rewards.mean().item():.4f}")
        print(f"  - Max Target Q-Value: {target_q_values.max().item():.4f}")
        print(f"  - Min Target Q-Value: {target_q_values.min().item():.4f}")
        print(f"  - Avg Target Q-Value: {target_q_values.mean().item():.4f}")
        print(f"  - Avg Current Q-Value: {current_q_values.mean().item():.4f}")
        print("-" * 40)

        return loss.item()  # Return the loss value


