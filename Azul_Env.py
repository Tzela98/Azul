from gym.spaces import Box, MultiDiscrete, Discrete
from azul_classes import GameLogic
import numpy as np

class AzulEnv:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.game_logic = GameLogic(num_players)
        self.state = self.game_logic.reset()
        self.current_player = self.state.current_player

        # Define observation space
        self.observation_space = Box(
            low=0,
            high=20,  # Adjust based on your game's logic
            shape=(64,),  # Size of the observation array
            dtype=np.float32
        )

        # Define a unified action space
        self.action_space = MultiDiscrete([
            2,  # source_type: 0 (factory) or 1 (center)
            5,  # source_idx: 0-4 (factories) or -1 (center)
            5,  # color: 0-4 (RED, BLUE, YELLOW, BLACK, WHITE)
            6   # target_row: 0-4 (rows) or 5 (floor)
        ])

        # Mapping between colors and indices
        self.color_to_index = {"RED": 0, "BLUE": 1, "YELLOW": 2, "BLACK": 3, "WHITE": 4}
        self.index_to_color = {v: k for k, v in self.color_to_index.items()}

    def reset(self):
        self.state = self.game_logic.reset()
        self.current_player = self.state.current_player
        return self.state.to_observation()

    def step(self, action):
        # Convert numerical action to the format expected by GameLogic
        if isinstance(action, (list, np.ndarray)) and len(action) == 4:
            # Action is in numerical format
            source_type = "factory" if action[0] == 0 else "center"
            source_idx = int(action[1])  # Ensure source_idx is an integer
            color = self.index_to_color.get(int(action[2]), None)  # Convert color index to string
            target_row = int(action[3]) if action[3] != 5 else -1  # Map 5 to -1 (floor)
            action_for_logic = (source_type, source_idx, color, target_row)
        elif isinstance(action, int):
            # Convert integer action (flattened index) to the unified format
            # Assuming the action space is 2 (source_type) * 5 (source_idx) * 5 (color) * 6 (target_row) = 300
            source_type = "factory" if (action // (5 * 5 * 6)) % 2 == 0 else "center"
            source_idx = (action // (5 * 6)) % 5
            color = self.index_to_color.get((action // 6) % 5, None)
            target_row = action % 6 if action % 6 != 5 else -1  # Map 5 to -1 (floor)
            action_for_logic = (source_type, source_idx, color, target_row)
        else:
            # Action is already in the format expected by GameLogic
            action_for_logic = action

        # Execute the action
        self.state, reward, done, info = self.game_logic.step(action_for_logic)
        self.current_player = self.state.current_player

        # Return the new observation, reward, done flag, and info
        return self.state.to_observation(), reward, done, info


    def get_valid_actions(self):
        current_player = self.state.players[self.current_player]
        valid_actions = self.game_logic.get_valid_actions(current_player)
        
        # Convert valid actions to numerical format
        numerical_actions = []
        for action in valid_actions:
            if isinstance(action, tuple):
                if len(action) == 3:  # Taking phase
                    source_type, source_idx, color = action
                    numerical_action = [
                        0 if source_type == "factory" else 1,  # source_type
                        source_idx if source_idx != -1 else 4,  # source_idx
                        self.color_to_index.get(color, 4),  # color
                        5  # Default to floor (target_row)
                    ]
                elif len(action) == 1:  # Placing phase
                    target_row = action[0]
                    numerical_action = [
                        0,  # Placeholder for source_type
                        0,  # Placeholder for source_idx
                        0,  # Placeholder for color
                        target_row if target_row != -1 else 5  # target_row
                    ]
                numerical_actions.append(numerical_action)
        
        return numerical_actions


