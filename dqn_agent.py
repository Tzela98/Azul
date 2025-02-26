import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture."""
    def __init__(self, input_size, output_size):
        super(DuelingQNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value for the state
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)  # Advantage for each action
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Ensure action is a list
        if isinstance(action, int):
            action = [action]  # Convert integer to a single-element list
        elif isinstance(action, (list, np.ndarray)):
            action = list(action)  # Ensure it's a list
        else:
            raise ValueError(f"Invalid action type: {type(action)}. Expected int, list, or np.ndarray.")

        # Store the experience as a tuple
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences
        batch = random.sample(self.buffer, batch_size)

        # Unpack the batch into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists to NumPy arrays for training
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Dueling DQN agent."""
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=1e-3, batch_size=64, buffer_capacity=10000, tau=0.005):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Initialize Q-network and target network
        self.input_size = env.observation_space.shape[0]
        self.output_size = 300  # Total number of actions (50 for taking phase + 6 for placing phase)
        self.q_network = DuelingQNetwork(self.input_size, self.output_size)
        self.target_network = DuelingQNetwork(self.input_size, self.output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def save_model(self, file_path: str):
        """Save the Q-network's state dictionary to a file."""
        torch.save(self.q_network.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str):
        """Load the Q-network's state dictionary from a file."""
        self.q_network.load_state_dict(torch.load(file_path))
        self.q_network.eval()
        print(f"Model loaded from {file_path}")

    def get_action(self, state):
        """Get a valid action from the agent's policy."""
        # Convert state to a PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

        # Get valid actions from the environment
        valid_actions = self.env.get_valid_actions()
        
        # Check if there are any valid actions
        if not valid_actions:
            raise ValueError("No valid actions available.")

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Randomly select a valid action
            action_idx = np.random.choice(len(valid_actions))
            action = valid_actions[action_idx]
        else:
            # Use the Q-network to select the best valid action
            with torch.no_grad():  # Disable gradient calculation for inference
                q_values = self.q_network(state_tensor)  # Get Q-values for all actions
                q_values = q_values.squeeze().numpy()  # Convert to NumPy array

            # Filter Q-values for valid actions
            valid_q_values = []
            for action in valid_actions:
                # Convert the action to an index (if necessary)
                action_idx = self._flatten_action(action)  # Use ReplayBuffer's flattening logic
                valid_q_values.append(q_values[action_idx])

            # Select the action with the highest Q-value among valid actions
            best_action_idx = np.argmax(valid_q_values)
            action = valid_actions[best_action_idx]

        # Ensure the action is a list
        if isinstance(action, int):
            action = [action]  # Convert integer to a single-element list
        elif isinstance(action, np.ndarray):
            action = action.tolist()  # Convert NumPy array to list

        return action

    def train(self):
        """Train the agent using a batch of experiences from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.q_network(states)
        current_q_values = current_q.gather(1, actions)  # Now both 2D tensors

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_actions = next_q.argmax(1).unsqueeze(1)  # Keep dims consistent
            next_q_values = next_q.gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()

        # Compute loss and update the Q-network
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
