import sys
from Azul_Classes import GameLogic, GameState
from tabulate import tabulate
from rich.console import Console
from rich.table import Table

console = Console()

class Game:
    def __init__(self, num_players=2):
        self.game_logic = GameLogic(num_players)
        self.state = self.game_logic.reset()
        self.num_players = num_players

    def display_game_state(self):
        console.print("\n[bold cyan]= GAME STATE =[/bold cyan]", style="bold underline")
        
        # Display Factories and Center Area in a table
        self.display_factories_and_center()
        
        # Display Player Boards
        for player in self.game_logic.players:
            self.display_player_board(player)
            console.print("-" * 40)

        print('ML friendly game state:')
        ML_friendly_state = self.state.to_observation()
        print(ML_friendly_state)
        print(len(ML_friendly_state))

    def display_factories_and_center(self):
        table = Table(title="Factories & Center Area", header_style="bold magenta")
        table.add_column("Factory #", justify="center")
        table.add_column("Tiles", justify="center")

        for i, factory in enumerate(self.game_logic.factories.factories):
            table.add_row(f"[cyan]{i}[/cyan]", f"{factory}")

        center_tiles = str(self.game_logic.central_area.tiles)
        center_token = "✔" if self.game_logic.central_area.has_first_player_token else "✘"
        table.add_row("[bold yellow]Center[/bold yellow]", f"{center_tiles} | Token: {center_token}")

        console.print(table)

    def display_player_board(self, player):
        console.print(f"\n[bold green]Player {player.player_id}[/bold green] (Score: [yellow]{player.score}[/yellow])")

        # Pattern Lines Table
        pattern_lines = [[f"Row {i+1}", str(line)] for i, line in enumerate(player.board.pattern_lines)]
        console.print(tabulate(pattern_lines, headers=["Pattern Line", "Tiles"], tablefmt="fancy_grid"))

        # Wall Display
        console.print("\n[bold]Wall:[/bold]")
        for row in player.board.wall_state:
            console.print(" ".join(["🟦" if cell else "⬜" for cell in row]))

        # Floor Line
        console.print(f"\n[red]Floor Line:[/red] {player.board.floor_line}")

    def get_human_action(self, player):
        valid_actions = self.game_logic.get_valid_actions(player)
        console.print("\n[bold cyan]Available Actions:[/bold cyan]")
        
        if self.game_logic.state.game_phase == "placing":
            console.print(f"Placing {self.game_logic.state.pending_count} "
                          f"{self.game_logic.state.pending_color} tiles")
        
        for i, action in enumerate(valid_actions):
            console.print(f"[bold yellow]{i}[/bold yellow]: {self.format_action(action)}")

        while True:
            choice = input("Choose action (0-{}): ".format(len(valid_actions)-1)).strip().lower()
            if choice == "exit":
                console.print("[bold red]Exiting the game...[/bold red]")
                sys.exit(0)  # Gracefully exit the game
            try:
                choice = int(choice)
                if 0 <= choice < len(valid_actions):
                    return valid_actions[choice]
                console.print("[red]Invalid choice![/red]")
            except ValueError:
                console.print("[red]Please enter a number or 'exit' to quit.[/red]")

    def format_action(self, action):
        if self.game_logic.state.game_phase == "taking":
            if action[2] is None:  # First player token case
                return "Take first player token from Center"
            source_type, source_idx, color = action
            source = f"Factory {source_idx}" if source_type == "factory" else "Center"
            return f"Take [bold]{color if color else 'FP token'}[/bold] from {source}"
        else:
            target_row = action[0]
            target = f"Row {target_row+1}" if target_row != -1 else "Floor"
            return f"Place on {target}"


    def play(self):
        while True:
            self.display_game_state()
            current_player = self.state.players[self.state.current_player]
            
            console.print(f"\n[bold blue]Player {current_player.player_id}'s turn:[/bold blue]")
            action = self.get_human_action(current_player)
            
            self.state, reward, done, _ = self.game_logic.step(action)
            
            if done:
                self.display_final_scores()
                break

    def display_final_scores(self):
        console.print("\n[bold red]GAME OVER![/bold red]")
        table = Table(title="Final Scores", header_style="bold cyan")
        table.add_column("Player", justify="center")
        table.add_column("Score", justify="center")

        for player in self.game_logic.players:
            table.add_row(f"Player {player.player_id}", f"[yellow]{player.score}[/yellow]")

        winner = max(self.game_logic.players, key=lambda p: p.score)
        console.print(table)
        console.print(f"[bold green]Winner: Player {winner.player_id} 🎉[/bold green]")

if __name__ == "__main__":
    num_players = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    game = Game(num_players)
    game.play()
