import numpy as np
from azul_env import AzulEnv
from dqn_agent import DQNAgent
import os
import logging  # Add this import

# Set up logging to a file
logging.basicConfig(
    filename="training_debug.log",  # Log file name
    level=logging.INFO,             # Log level (INFO, DEBUG, etc.)
    format="%(asctime)s - %(message)s",  # Log format
    filemode="w"                    # Overwrite the log file each time
)


def train_agent(num_episodes=10000, num_players=2, save_model_path="azul_dqn.pth", save_interval=1000):
    """
    Train the DQN agent using self-play (agent plays against itself).

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

        logging.info(f"\n=== Starting Episode {episode + 1} ===")

        while not done:
            # Get the current player's action
            if env.state.current_player == 0:
                # Agent's turn: get action and unflatten it
                action = agent.get_action(state)

                # Debug: Log the action
                logging.info(f"\nAgent's Turn (Player 0):")
                logging.info(f"  State: {state}")
                logging.info(f"  Selected Action: {action}")

                # Ensure action is a single list or NumPy array with 4 elements
                if isinstance(action, tuple):
                    action = np.concatenate(action)  # Combine all arrays into one
                if isinstance(action, np.ndarray):
                    action = action.tolist()  # Convert to list if necessary
                if not isinstance(action, list) or len(action) != 4:
                    logging.error(f"Action must be a list or NumPy array with 4 elements, but got {type(action)}: {action}")
                    raise ValueError(f"Action must be a list or NumPy array with 4 elements, but got {type(action)}: {action}")

                # Check if the selected action is valid
                valid_actions = env.get_valid_actions()
                logging.info(f"Valid Actions: {valid_actions}")
                logging.info(f"Selected Action: {action}")
                if action not in valid_actions:
                    logging.error(f"Invalid Action: {action}")
                    raise ValueError("Invalid action selected by the agent.")

                # Take a step in the environment
                next_state, rewards, done, info = env.step(action)

                # Debug: Log the results of the step
                logging.info(f"  Next State: {next_state}")
                logging.info(f"  Rewards: {rewards}")
                logging.info(f"  Done: {done}")
                logging.info(f"  Info: {info}")

                # Store the experience in the replay buffer (only for the agent)
                agent.replay_buffer.push(state, action, rewards, next_state, done)

                # Train the agent
                loss = agent.train()

                # Update state and total reward
                state = next_state
                total_reward += rewards
            else:
                # Opponent's turn: use the latest version of the agent to decide the opponent's move
                action = agent.get_action(state)

                # Debug: Log the action
                logging.info(f"\nOpponent's Turn (Player {env.state.current_player}):")
                logging.info(f"  State: {state}")
                logging.info(f"  Selected Action: {action}")

                # Ensure action is a single list or NumPy array with 4 elements
                if isinstance(action, tuple):
                    action = np.concatenate(action)  # Combine all arrays into one
                if isinstance(action, np.ndarray):
                    action = action.tolist()  # Convert to list if necessary
                if not isinstance(action, list) or len(action) != 4:
                    logging.error(f"Action must be a list or NumPy array with 4 elements, but got {type(action)}: {action}")
                    raise ValueError(f"Action must be a list or NumPy array with 4 elements, but got {type(action)}: {action}")

                # Check if the selected action is valid
                valid_actions = env.get_valid_actions()
                logging.info(f"Valid Actions: {valid_actions}")
                logging.info(f"Selected Action: {action}")
                if action not in valid_actions:
                    logging.error(f"Invalid Action: {action}")
                    raise ValueError("Invalid action selected by the agent.")

                # Take a step in the environment for the opponent
                next_state, rewards, done, info = env.step(action)

                # Debug: Log the results of the step
                logging.info(f"  Next State: {next_state}")
                logging.info(f"  Rewards: {rewards}")
                logging.info(f"  Done: {done}")
                logging.info(f"  Info: {info}")

                # Update state
                state = next_state

        # Log episode results
        logging.info(f"\n=== Episode {episode + 1} Summary ===")
        logging.info(f"  Total Reward: {total_reward}")
        logging.info(f"  Epsilon: {agent.epsilon:.4f}")
        if loss is not None:
            logging.info(f"  Loss: {loss:.4f}")

        # Save the model periodically
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"{save_model_path}_episode_{episode + 1}.pth")
            agent.save_model(model_path)
            logging.info(f"\nModel saved to {model_path}")

    # Save the final model
    final_model_path = os.path.join("models", save_model_path)
    agent.save_model(final_model_path)
    logging.info(f"\nTraining complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    # Train the agent
    train_agent(num_episodes=1000, num_players=2, save_model_path="final_model.pth")