from env.board import Board
from agents.player import Player, RandomPlayer, Agent


class Gomoku:
    def __init__(self, board: Board):
        self.board = board

    def assign_players(self, player1: Player, player2: Player) -> dict[int, Player]:
        player1.set_player_id(1)
        player2.set_player_id(2)
        players = {1: player1, 2: player2}
        self.players = players
        return self.players

    def step(self, move: int) -> tuple[bool, int]:
        self.board.play_move(move)
        return self.board.is_game_end()

    def start_play(
        self,
        agent: Agent,
        random_player: RandomPlayer,
        start_player: int = 1,
    ):
        players = {1: agent, 2: random_player}
        agent.set_player_id(1)
        random_player.set_player_id(2)

        self.board.init_board(start_player)
        while True:
            current_player_id = self.board.get_current_player()
            current_player = players[current_player_id]
            move = current_player.get_action(self.board)

            self.board.play_move(move)
            agent.mcts.update(move)

            is_end, winner = self.board.is_game_end()
            if is_end:
                return winner

    def start_self_play(
        self,
        player: Agent,
        start_player: int = 1,
        is_shown: bool = False,
        temp: float = 1e-3,
    ):
        self.board.init_board(start_player)
        # TODO:

    def _print_result(self, winner: int, players: dict[int, Player]) -> None:
        if winner == -1:
            print("Game ended. Draw.")
        else:
            symbol = "X" if winner == 1 else "O"
            print(f"Game ended. Winner is Player {winner} ({symbol})")
