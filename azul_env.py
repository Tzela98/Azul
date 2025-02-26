from gym.spaces import Box, MultiDiscrete
from azul_classes import GameLogic
import numpy as np
import logging
import random

class AzulEnv:
    def __init__(self, num_players=2):
        self.logger = logging.getLogger('AzulEnv')
        self.num_players = num_players
        self.game = GameLogic(num_players)
        self.observation_space = Box(low=0, high=20, shape=(64,), dtype=np.float32)
        self.action_space = MultiDiscrete([2, 5, 5, 6])  # [source_type, source_idx, color, target_row]
        self.color_map = {"RED":0, "BLUE":1, "YELLOW":2, "BLACK":3, "WHITE":4}
        self.pending_action = None

    def reset(self):
        """Reset the environment and return the initial observation."""
        self.logger.info("Resetting environment")
        self.game.reset()
        self.pending_action = []
        return self._get_observation()

    def _get_observation(self):
        """Convert the game state into an observation vector."""
        try:
            return np.array(self.game.state.to_observation(), dtype=np.float32)
        except AttributeError:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, combined_action):
        """Execute one step in the environment."""
        try:
            reward = 0
            done = False
            
            # Log the action in human-readable format
            self.logger.info(f"Attempting action: {self.translate_action(combined_action)}")
            
            if self.game.state.game_phase == "taking":
                self._handle_take_phase(combined_action[:3])
                return self._get_observation(), 0, False, {}
                
            elif self.game.state.game_phase == "placing":
                reward = self._handle_place_phase(combined_action[3])
                done = self.game._check_game_over()
                
                # Reset pending_action after placing phase
                self.pending_action = None
                return self._get_observation(), reward, done, {}

        except Exception as e:
            self.logger.error(f"Invalid action: {self.translate_action(combined_action)} | Error: {str(e)}")
            return self._get_observation(), -10, True, {}
    
    def translate_action(self, action):
        """Convert numerical action to human-readable format."""
        source_map = {0: "Factory", 1: "Center"}
        color_map = {0: "Red", 1: "Blue", 2: "Yellow", 3: "Black", 4: "White"}
        
        source_type = source_map[action[0]]
        # Use 1-based numbering ONLY for factories
        source_idx = action[1] + 1 if source_type == "Factory" else 0
        color = color_map[action[2]]
        target = "Floor" if action[3] == 5 else f"Row {action[3] + 1}"
        
        return f"Take {color} from {source_type} {source_idx} → Place on {target}"

    def _handle_take_phase(self, take_action):
        """Execute tile-taking action with enhanced validation."""
        source_type = "factory" if take_action[0] == 0 else "center"
        source_idx = int(take_action[1])
        color_idx = take_action[2]
        
        try:
            color = list(self.color_map.keys())[color_idx]
        except IndexError:
            raise ValueError(f"Invalid color index {color_idx}")
        
        current_player = self.get_current_player()
        all_valid = self.game.get_valid_actions(current_player)

        self.logger.info(f"Checking action: Source Type={source_type}, Source Index={source_idx}, Color={color} (Index={color_idx})")
        self.logger.info(f"Valid actions from env: {all_valid}")

        for a in all_valid:
            self.logger.info(f"Valid Action: Source Type={a[0]}, Source Index={a[1]}, Color={a[2]}")

        
        valid = next(
            (a for a in all_valid 
             if a[0] == source_type 
             and a[1] == source_idx 
             and (a[2] == color or (a[2] is None and color_idx == 4))),
            None
        )

        self.logger.info(f"Matched valid action: {valid}")
        
        if not valid:
            # Log valid actions in human-readable format
            valid_actions = [self.translate_action([0 if a[0] == "factory" else 1, a[1], self.color_map.get(a[2], 4), 5]) 
                           for a in all_valid]
            self.logger.error(f"Action not valid. Valid actions: {valid_actions}")
            raise ValueError("Invalid take action")
        
        self.pending_action = take_action
        self.game.step(valid)

    def _handle_place_phase(self, target_row):
        """Execute tile-placing action."""
        place_action = (target_row if target_row < 5 else -1,)
        current_player = self.get_current_player()
        valid_actions = self.game.get_valid_actions(current_player)
        
        if place_action not in valid_actions:
            raise ValueError(f"Invalid place action: {place_action}")
        
        prev_score = current_player.score
        self.game.step(place_action)
        return (current_player.score - prev_score) / 10.0

    def get_valid_actions(self):
        """Generate valid combined actions for current phase, based on game logic."""
        valid = []
        phase = self.game.state.game_phase
        current_player = self.get_current_player()
        
        if phase == "taking":
            # Get valid take actions from game logic
            take_actions = self.game.get_valid_actions(current_player)
            
            # Convert to numerical format
            for action in take_actions:
                source_type = 0 if action[0] == "factory" else 1
                source_idx = action[1] if action[1] != -1 else 0  # Map -1 to 0 for center
                color_idx = self.color_map.get(action[2], 4) if action[2] else 4  # Handle None
                valid.append([source_type, source_idx, color_idx, 5])  # Placeholder for place phase
        
        elif phase == "placing":

            # Ensure pending_action is not None
            if self.pending_action is None:
                self.logger.warning("No pending action found during placing phase.")
                return []
            
            # Get valid place actions from game logic
            place_actions = self.game.get_valid_actions(current_player)
            
            # Preserve the original take action
            take_part = self.pending_action[:3]
            for action in place_actions:
                valid.append([
                    take_part[0],  # source_type
                    take_part[1],  # source_idx
                    take_part[2],  # color_idx
                    action[0] if action[0] != -1 else 5  # target_row
                ])
        
        return valid
    
    def sample_valid_action(self):
        """
        Samples a valid action from the list of allowed actions based on the current game state.
        Returns a valid action in the format [source_type, source_idx, color, target_row].
        """
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            raise ValueError("No valid actions available in the current game state.")
        return random.choice(valid_actions)


    def get_current_player(self):
        """Get the current player object from game state."""
        return self.game.state.players[self.game.state.current_player]
