import random
from collections import Counter
import numpy as np


class TileBag:
    def __init__(self):
        self.colors = ['RED', 'BLUE', 'YELLOW', 'BLACK', 'WHITE']
        self.tiles = []
        for color in self.colors:
            self.tiles.extend([color] * 20)
        random.shuffle(self.tiles)
        self.discard_pile = []

    def draw_tiles(self, num_tiles: int) -> list[str]:
        drawn = []
        while len(drawn) < num_tiles and self.tiles:
            drawn.append(self.tiles.pop())

        if len(drawn) < num_tiles and self.discard_pile:
            self.refill_from_discard()
            while len(drawn) < num_tiles and self.tiles:
                drawn.append(self.tiles.pop())

        return drawn
    
    def discard_tiles(self, tiles: list[str]) -> None:
        self.discard_pile.extend(tiles)

    def refill_from_discard(self) -> None:
        self.tiles.extend(self.discard_pile)
        random.shuffle(self.tiles)
        self.discard_pile = []

    def remaining_tiles_count(self) -> dict[str, int]:
        return dict(Counter(self.tiles))
    

class FactoryDisplay:
    def __init__(self, num_factories: int):
        self.num_factories = num_factories
        self.factories = [[] for _ in range(num_factories)]

    def reset(self, tile_bag: TileBag) -> None:
        for i in range(self.num_factories):
            self.factories[i] = tile_bag.draw_tiles(4)

    def take_tiles(self, factory_idx: int, color: str) -> tuple[list[str], list[str]]:
        factory = self.factories[factory_idx]

        taken =  [t for t in factory if t == color]
        remaining = [t for t in factory if t!= color]

        self.factories[factory_idx] = []

        return taken, remaining
    
    def is_empty(self) -> bool:
        return all(len(factory) == 0 for factory in self.factories)
    
    def state_repr(self) -> dict:
        return {
            'factories': [f.copy() for f in self.factories],
            'num_factories': self.num_factories
        }
    
class CentralArea:
    def __init__(self):
        self.tiles = []
        self.has_first_player_token = False

    # Prepare for new round
    def reset(self) -> None:
        self.tiles = []
        self.has_first_player_token = True

    def add_tiles(self, tiles: list[str]) -> None:
        self.tiles.extend(tiles)

    def take_tiles(self, color: str) -> tuple[list[str], bool]:
        if color not in self.tiles:
            return [], False
        
        taken_tiles = [t for t in self.tiles if t == color]
        self.tiles = [t for t in self.tiles if t!= color]
        took_token = self.has_first_player_token

        if took_token:
            self.has_first_player_token = False

        return taken_tiles, took_token
    
    def get_available_colors(self) -> list[str]:
        return list(set(self.tiles))
    
    def state_repr(self) -> dict:
        return {
            'tiles': self.tiles.copy(),
            'has_token': self.has_first_player_token
        }
    
    def __str__(self):
        return f"CentralArea(tiles={self.tiles}, token={self.has_first_player_token})"
    

class PlayerBoard:
    def __init__(self):
        self.pattern_lines = [[] for _ in range(5)]
        self.wall = [
            ["BLUE", "YELLOW", "RED", "BLACK", "WHITE"],  # Row 0 valid colors
            ["WHITE", "BLUE", "YELLOW", "RED", "BLACK"],  # Row 1
            ["BLACK", "WHITE", "BLUE", "YELLOW", "RED"],   # Row 2
            ["RED", "BLACK", "WHITE", "BLUE", "YELLOW"],   # Row 3
            ["YELLOW", "RED", "BLACK", "WHITE", "BLUE"]    # Row 4
        ]
        # Initializes a 5x5 grid with values false
        self.wall_state = [[False for _ in range(5)] for _ in range(5)]
        self.floor_line = []
        self.score = 0

    def row_completed(self, row_idx: int) -> bool:
        max_capacity = row_idx + 1
        return len(self.pattern_lines[row_idx]) == max_capacity

    def place_tiles(self, color: str, row_idx: int, num_tiles: int) -> list:
        max_capacity = row_idx + 1
        current_tiles = len(self.pattern_lines[row_idx])

        # Reject if row has conflicting color
        if current_tiles > 0 and self.pattern_lines[row_idx][0] != color:
            self.floor_line.extend([color] * num_tiles)
            return [color] * num_tiles  # Return the overflowed tiles

        available_space = max_capacity - current_tiles
        tiles_to_place = min(available_space, num_tiles)
        overflow = num_tiles - tiles_to_place

        # Update pattern line and return overflow
        self.pattern_lines[row_idx].extend([color] * tiles_to_place)
        self.floor_line.extend([color] * overflow)

        return [color] * overflow  # Ensure a list is always returned

    def score_round(self) -> int:
        round_points = 0
        for row_idx, pattern_line in enumerate(self.pattern_lines):
            if len(pattern_line) == row_idx + 1:  # Full pattern line
                color = pattern_line[0]
                col_idx = self.wall[row_idx].index(color)
                
                # Add to wall
                self.wall_state[row_idx][col_idx] = True
                self.pattern_lines[row_idx] = []
                
                # Calculate score for this tile
                round_points += self._calculate_wall_score(row_idx, col_idx)
        
        return round_points
    
    def _calculate_wall_score(self, row: int, col: int) -> int:
        # Horizontal and vertical directions
        left = self._count_continuous(row, col, 0, -1)
        right = self._count_continuous(row, col, 0, 1)
        up = self._count_continuous(row, col, -1, 0)
        down = self._count_continuous(row, col, 1, 0)
        
        # Sum adjacent tiles + 1 for current tile
        h_neighbors = left + right
        v_neighbors = up + down
        total = 1
        
        if h_neighbors > 0:
            total += h_neighbors
        if v_neighbors > 0:
            total += v_neighbors
            
        return total
    
    def _count_continuous(self, row: int, col: int, dr: int, dc: int) -> int:
        # Count consecutive tiles in direction (dr, dc)
        count = 0
        r, c = row + dr, col + dc
        
        while 0 <= r < 5 and 0 <= c < 5:
            if self.wall_state[r][c]:
                count += 1
                r += dr
                c += dc
            else:
                break
        return count
    
    def apply_floor_penalties(self) -> int:
        # Calculate and reset floor line penalties
        penalty_scores = [-1, -1, -2, -2, -2, -3, -3]  # Azul's penalty rules
        num_penalties = min(len(self.floor_line), 7)
        penalty = sum(penalty_scores[:num_penalties])
        self.score += penalty
        self.floor_line = []
        return penalty

    def state_repr(self) -> dict:
        # RL-friendly state representation
        return {
            "pattern_lines": [len(line) for line in self.pattern_lines],
            "wall": [[int(cell) for cell in row] for row in self.wall_state],
            "floor_count": len(self.floor_line),
            "score": self.score
        }
    

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.board = PlayerBoard()
        self.score = 0
        self.has_first_player_token = False

    def reset_for_new_round(self) -> None:
        # Reset temporary state between rounds
        self.board.floor_line = []
        self.has_first_player_token = False

    def add_score(self, points: int) -> None:
        # Update player score (positive or negative)
        self.score += points

    def take_first_player_token(self) -> None:
        # Set first player token status
        self.has_first_player_token = True

    def get_observation(self) -> dict:
        # Return RL-observable player state (excluding hidden info)
        return {
            "score": self.score,
            "has_first_token": self.has_first_player_token,
            **self.board.state_repr()
        }

    def __repr__(self) -> str:
        return f"Player {self.player_id} (Score: {self.score})"
    

class GameState:
    def __init__(self, players, factories, central_area, tile_bag, current_player=0):
        # Core game components
        self.players = players  
        self.factories = factories  
        self.central_area = central_area  
        self.tile_bag = tile_bag  
        
        # Game progression tracking
        self.current_player = current_player  
        self.game_phase = "taking"  # "taking" or "placing"
        self.round_number = 0
        
        # Pending state for tile placement
        self.pending_color = None  # Color of tiles to be placed
        self.pending_count = 0    # Number of tiles to be placed
        self.pending_source = None  # Source of tiles ("factory" or "center")

    def to_observation(self):
        # Convert game state to numerical array for RL
        obs = []
        
        # Add factory states
        for factory in self.factories.factories:
            obs += self._one_hot_tiles(factory)
            
        # Add central area state
        obs += self._one_hot_tiles(self.central_area.tiles)
        obs.append(1 if self.central_area.has_first_player_token else 0)
        
        # Add current player's board state
        current_player = self.players[self.current_player]
        obs += self._encode_player_state(current_player)
        
        # Add pending state
        obs.append(self.pending_count if self.pending_color else 0)
        obs.append(self._color_to_index(self.pending_color) if self.pending_color else -1)
        
        # Convert to numpy array
        return np.array(obs)
    
    def _one_hot_tiles(self, tiles):
        # Create 5-element one-hot vector (for 5 colors)
        color_map = {"RED":0, "BLUE":1, "YELLOW":2, "BLACK":3, "WHITE":4}
        counts = [0]*5
        for tile in tiles:
            counts[color_map[tile]] += 1
        return counts
    
    def _encode_player_state(self, player):
        # Encode pattern lines, wall, and floor
        encoded = []
        # Pattern line lengths
        encoded += [len(line) for line in player.board.pattern_lines]
        # Wall tiles (flattened)
        encoded += [tile for row in player.board.wall_state for tile in row]
        # Floor line count
        encoded.append(len(player.board.floor_line))
        return encoded
    
    def _color_to_index(self, color):
        # Map color to index for observation
        color_map = {"RED":0, "BLUE":1, "YELLOW":2, "BLACK":3, "WHITE":4}
        return color_map.get(color, -1)


class GameLogic:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.tile_bag = TileBag()
        self.factories = FactoryDisplay(5 if num_players == 2 else 9)
        self.central_area = CentralArea()
        self.players = [Player(i) for i in range(num_players)]
        self.state = None

    def reset(self):
        self.tile_bag = TileBag()
        self.factories.reset(self.tile_bag)
        self.central_area.reset()
        for player in self.players:
            player.reset_for_new_round()
        self.state = GameState(
            players=self.players,
            factories=self.factories,
            central_area=self.central_area,
            tile_bag=self.tile_bag,
            current_player=0
        )
        return self.state

    def get_valid_actions(self, player):
        valid_actions = []
        
        if self.state.game_phase == "taking":
            # Factory sources
            for f_idx, factory in enumerate(self.factories.factories):
                if not factory:
                    continue
                for color in set(factory):
                    valid_actions.append(("factory", f_idx, color))
            
            # Center source
            if self.central_area.tiles or self.central_area.has_first_player_token:
                colors = set(self.central_area.tiles) if self.central_area.tiles else set()
                if colors:
                    for color in colors:
                        valid_actions.append(("center", -1, color))
                else:
                    # Allow taking first player token even if no tiles
                    valid_actions.append(("center", -1, None))
        
        elif self.state.game_phase == "placing":
            # Get valid target rows for pending color
            valid_rows = self._get_valid_target_rows(player, self.state.pending_color)
            valid_actions = [(row,) for row in valid_rows]
        
        return valid_actions

    def _get_valid_target_rows(self, player, color):
        valid_rows = []
        for row_idx in range(5):
            line = player.board.pattern_lines[row_idx]
            if len(line) < row_idx + 1 and (not line or line[0] == color):  # Allow placing if row is empty or matches color and is not full
                valid_rows.append(row_idx)
        valid_rows.append(-1)  # Always allow placement on the floor line
        return valid_rows

    def step(self, action):
        current_player = self.state.players[self.state.current_player]
        prev_wall = [row.copy() for row in current_player.board.wall_state]
        prev_score = current_player.score
        target_row = None
        took_token = None

        if self.state.game_phase == "taking":
            # Handle tile taking
            if len(action) == 4:
                # Unified action format
                source_type, source_idx, color, _ = action  # Ignore target_row for taking phase
            else:
                # Legacy action format (for backward compatibility)
                source_type, source_idx, color = action
            
            # Take tiles from source
            if source_type == "factory":
                taken, remaining = self.factories.take_tiles(source_idx, color)
                self.central_area.add_tiles(remaining)
            else:
                taken, took_token = self.central_area.take_tiles(color)
                if took_token:
                    current_player.take_first_player_token()
            
            # Update pending state
            self.state.pending_color = color
            self.state.pending_count = len(taken)
            self.state.pending_source = source_type
            self.state.game_phase = "placing"
        
        elif self.state.game_phase == "placing":
            # Handle tile placement
            if len(action) == 4:
                # Unified action format
                target_row = action[3]  # Use target_row for placing phase
            else:
                # Legacy action format (for backward compatibility)
                target_row = action[0]
            
            # Place tiles on pattern line or floor
            if target_row == -1:
                current_player.board.floor_line.extend([self.state.pending_color] * self.state.pending_count)
            else:
                overflow = current_player.board.place_tiles(self.state.pending_color, target_row, self.state.pending_count)
                current_player.board.floor_line.extend(overflow)
            
            # Reset pending state
            self.state.pending_color = None
            self.state.pending_count = 0
            self.state.pending_source = None
            
            # Advance to next player
            self.state.current_player = (self.state.current_player + 1) % self.num_players
            self.state.game_phase = "taking"
            
            # Check round completion
            if self.factories.is_empty() and not self.central_area.tiles:
                self._score_round()

        # Calculate reward and check termination
        game_over = self._check_game_over()
        is_winner = self.is_winner(current_player) if game_over else False
        reward = self.calculate_reward(
            current_player=current_player,
            prev_score=prev_score,
            prev_wall=prev_wall,
            took_token=took_token,
            target_row=target_row,
            game_over=game_over,
            is_winner=is_winner
        )
        
        return self.state, reward, game_over, {}
    

    def calculate_reward(self, current_player, prev_score, prev_wall, took_token, target_row, game_over=False, is_winner=False):
        reward = 0 

        # 1. Immediate Score Change (normalized)
        reward += (current_player.board.score - prev_score) / 10.0

        # 2. Reward for completing a row
        for row_idx in range(5):
            if current_player.board.row_completed(row_idx):
                reward += 1.0

        # 3. Penalty for placing tiles in the floor line
        if target_row == -1:
            reward -= 1.0

        # 4. Reward for progress on the wall
        new_wall = current_player.board.wall_state
        for i in range(5):
            for j in range(5):
                if new_wall[i][j] and not prev_wall[i][j]:
                    reward += 0.5

        # 5. Reward for taking the first player token
        if took_token:
            reward += 1.0

        # 6. Reward for winning the game
        if game_over and is_winner:
            reward += 10

        return reward
    
    def is_winner(self, player):
        """
        Check if the given player is the winner.
        """
        # Compare the player's score to other players' scores
        max_score = max(p.board.score for p in self.state.players)
        return player.board.score == max_score

    def _score_round(self):
        # Score all players
        for player in self.players:
            # Score pattern lines
            round_points = player.board.score_round()
            player.add_score(round_points)
            
            # Apply floor penalties
            penalty = player.board.apply_floor_penalties()
            player.add_score(penalty)
        
        # Start new round
        self._start_new_round()


    def _start_new_round(self):
        # Refill factories
        self.factories.reset(self.tile_bag)
        
        # Reset central area with first player token
        self.central_area.reset()
        
        # Reset player boards
        for player in self.players:
            player.reset_for_new_round()
        
        # Determine first player
        new_first_player = 0
        for idx, player in enumerate(self.players):
            if player.has_first_player_token:
                new_first_player = idx
                player.has_first_player_token = False
                break
        
        # Update game state
        self.state.current_player = new_first_player
        self.state.game_phase = "taking"

    def _check_game_over(self):
        # Check for completed wall row
        for player in self.players:
            for row in player.board.wall_state:
                if all(row):
                    # Game ends if any player completes a row, but only after the current round is completed
                    return True
        
        # Check if the tile bag is empty, no more tiles can be drawn, and the current round is completed
        if not self.tile_bag.tiles and not self.tile_bag.discard_pile:
            # Ensure all factories and the center are empty (current round is completed)
            if self.factories.is_empty() and not self.central_area.tiles:
                return True  # Game ends if no more tiles are available and the round is completed
        
        return False
