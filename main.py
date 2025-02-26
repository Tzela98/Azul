import numpy as np
from azul_env import AzulEnv
from dqn_agent import DQNAgent
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f"training_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

def train_agent(num_episodes=1000, num_players=2, save_model_path="azul_dqn.pth", save_interval=100):
    env = AzulEnv(num_players=num_players)
    agent = DQNAgent(
        env,
        batch_size=64,
        epsilon_decay=0.997,
        epsilon_min=0.02,
        lr=1e-4,
        tau=0.01
    )

    # Initialize metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'buffer_sizes': []
    }

    # Pre-fill replay buffer
    logging.info("Initializing replay buffer...")
    while len(agent.replay_buffer) < agent.batch_size:
        state = env.reset()
        done = False
        while not done:
            action = env.sample_valid_action()
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
    logging.info(f"Initial buffer size: {len(agent.replay_buffer)}")

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        
        logging.info(f"\n=== Episode {episode+1}/{num_episodes} ===")
        
        while not done:
            try:
                # Get action from agent
                action = agent.get_action(state)
                
                # Log the action in human-readable format
                logging.info(f"Attempting action: {env.translate_action(action)}")
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train agent
                if len(agent.replay_buffer) >= agent.batch_size:
                    loss = agent.train()
                    episode_losses.append(loss)
                    logging.debug(f"Step loss: {loss:.4f}")
                
                total_reward += reward
                state = next_state
                
            except Exception as e:
                logging.error(f"Episode {episode} failed: {str(e)}")
                break

        # Update metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        metrics['rewards'].append(total_reward)
        metrics['losses'].append(avg_loss)
        metrics['epsilons'].append(agent.epsilon)
        metrics['buffer_sizes'].append(len(agent.replay_buffer))

        # Log episode summary
        logging.info(
            f"Episode {episode+1} | "
            f"Reward: {total_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Buffer: {len(agent.replay_buffer)}"
        )

        # Save model periodically
        if (episode + 1) % save_interval == 0:
            save_path = f"models/{save_model_path}_ep{episode+1}.pth"
            agent.save_model(save_path)
            logging.info(f"Model saved to {save_path}")

        # Visualize training progress
        if (episode + 1) % 50 == 0:
            _plot_progress(metrics)

    # Final save
    agent.save_model(f"models/{save_model_path}_final.pth")
    logging.info("Training completed successfully")
    return metrics

def _plot_progress(metrics):
    """Helper function to plot training metrics"""
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Reward plot
    plt.subplot(2, 2, 1)
    plt.plot(metrics['rewards'])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(metrics['losses'])
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    
    # Epsilon plot
    plt.subplot(2, 2, 3)
    plt.plot(metrics['epsilons'])
    plt.title("Exploration Rate")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    
    # Buffer size plot
    plt.subplot(2, 2, 4)
    plt.plot(metrics['buffer_sizes'])
    plt.title("Replay Buffer Size")
    plt.xlabel("Episode")
    plt.ylabel("Buffer Size")
    
    plt.tight_layout()
    plt.savefig("plots/training_progress.png")
    plt.close()

def test_legal_moves(env):
    """Test that the agent only selects legal moves."""
    env.reset()
    print("Testing legal moves...")
    
    # Get valid actions
    valid_actions = env.get_valid_actions()
    print("Valid actions:")
    for action in valid_actions:
        print(f"  {env.translate_action(action)}")
    
    # Ensure the agent only selects valid actions
    for _ in range(10):
        action = env.sample_valid_action()
        
        # Check if action is in valid_actions
        is_valid = any(np.array_equal(action, valid_action) for valid_action in valid_actions)
        
        if not is_valid:
            print(f"ERROR: Invalid action selected: {env.translate_action(action)}")
        else:
            print(f"Valid action selected: {env.translate_action(action)}")

if __name__ == "__main__":
    env = AzulEnv(num_players=2)
    test_legal_moves(env)
    train_agent(num_episodes=1000, num_players=2)
