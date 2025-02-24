import numpy as np
from azul_env import AzulEnv
from dqn_agent import DQNAgent

def train_agent(num_episodes=1000, num_players=2, save_model_path="azul_dqn.pth", save_interval=100):
    """
    Train the DQN agent by playing against itself.

    Args:
        num_episodes (int): Number of episodes to train.
        num_players (int): Number of players in the game.
        save_model_path (str): Path to save the trained model.
        save_interval (int): Save the model every N episodes.
    """
    # Initialize the environment and agent
    env = AzulEnv(num_players=num_players)
    agent = DQNAgent(env)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get action for the current player
            action = agent.get_action(state)

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)

            # Store the experience in the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train the agent
            loss = agent.train()

            # Update state and total reward
            state = next_state
            total_reward += reward

        # Print episode results
        print(f"Episode {episode + 1}")
        print(f"  Total Reward: {total_reward}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        if loss is not None:
            print(f"  Loss: {loss:.4f}")

        # Save the model periodically
        if (episode + 1) % save_interval == 0:
            agent.save_model(f"{save_model_path}_episode_{episode + 1}.pth")
            print(f"Model saved to {save_model_path}_episode_{episode + 1}.pth")

    # Save the final model
    agent.save_model(save_model_path)
    print(f"Training complete. Final model saved to {save_model_path}")


if __name__ == "__main__":
    # Train the agent
    train_agent(num_episodes=1000, num_players=2, save_model_path="azul_dqn.pth")