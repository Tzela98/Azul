import gym
import torch
import random
import numpy as np
from Azul_Env import AzulEnv  # Import the custom Azul environment
from DQN import DQNAgent  # Import the Deep Q-Network (to be implemented)

def train_agent(env, episodes=1000):
    """Trains a Deep Q-Learning agent."""
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    return agent

def self_play(env, agent, episodes=10):
    """Runs self-play games with the trained agent."""
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, training=False)  # No exploration
            state, _, done, _ = env.step(action)

        env.render()  # Display final game state

if __name__ == "__main__":
    env = AzulEnv(num_players=2)
    
    print("Training DQN Agent...")
    trained_agent = train_agent(env, episodes=1000)
    
    print("\nRunning Self-Play...")
    self_play(env, trained_agent, episodes=5)

    env.close()
