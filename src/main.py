from env.board import Board
from env.gomoku import Gomoku


def mcts_and_random(board_size: int, n_in_a_row: int, n_simulations: int, n_games: int):
    board = Board(board_size, n_in_a_row)
    game = Gomoku(board)
