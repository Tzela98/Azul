import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """Neural network for Deep Q-Learning."""
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output raw Q-values (no activation)

class ReplayBuffer:
    """Experience replay buffer to store and sample experiences."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for playing Azul using Q-learning with experience replay."""
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay=0.995):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon  # Exploration rate
        self.min_epsilon = min_epsilon
        self.decay = decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network and Target Network
        self.q_network = QNetwork(input_dim, output_dim).to(self.device)
        self.target_network = QNetwork(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Sync networks
        self.target_network.eval()  # Target network doesn't train

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer()

    def select_action(self, state, training=True):
        """Selects an action using an epsilon-greedy strategy."""
        if training and random.random() < self.epsilon:
            return (random.randint(0, 2), random.randint(0, 5), random.randint(0, 4))  # Sample valid tuple


        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def train(self, batch_size=64):
        """Trains the DQN using experience replay."""
        if self.memory.size() < batch_size:
            return  # Wait until we have enough experiences

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        actions = actions.argmax(dim=1).long().unsqueeze(1)  # Shape will be (batch_size, 1)

        # Compute current Q-values
        print(f"States shape: {states.shape}")  # Should be (batch_size, state_dim)
        print(f"Actions shape: {actions.shape}")  # Should be (batch_size, 1)
        print(f"Q-values shape: {self.q_network(states).shape}")  # Should be (batch_size, num_actions)
        
        current_q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def update_target_network(self):
        """Updates the target network to match the Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
