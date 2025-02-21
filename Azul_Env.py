import gym
import numpy as np
from gym import spaces
from Azul_Classes import GameLogic  # Import your game logic

class AzulEnv(gym.Env):
    """Custom Gym environment for Azul."""
    
    def __init__(self, num_players=2):
        super(AzulEnv, self).__init__()
        
        self.num_players = num_players
        self.game_logic = GameLogic(num_players)
        
        # Define the observation space (ML-friendly state representation)
        state_shape = self.game_logic.reset().to_observation()
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(state_shape),), dtype=np.float32)
        
        # Define the action space (Discrete for simplicity)
        self.action_space = spaces.Discrete(len(self.game_logic.get_valid_actions(self.game_logic.state.players[0])))

    def reset(self):
        """Resets the game and returns the initial observation."""
        self.state = self.game_logic.reset()
        return np.array(self.state.to_observation(), dtype=np.float32)

    def step(self, action):
        """Executes an action in the game and returns the new state, reward, done, and additional info."""
        current_player = self.state.players[self.state.current_player]
        
        # Execute action
        next_state, reward, done, _ = self.game_logic.step(action)
        
        # Convert state to ML-friendly representation
        obs = np.array(next_state.to_observation(), dtype=np.float32)
        
        return obs, reward, done, {}

    def render(self, mode="human"):
        """Optional: Prints the game state for debugging."""
        self.game_logic.display_game_state()

    def close(self):
        """Cleans up the environment."""
        pass
