import numpy as np


class Board:
    """
    Gomoku board with RL-friendly state representation.

    Board is represented as an integer array comprising each cell values of:
        0 = empty, 1 = player1, 2 = player2

    Order of the play: player1 goes first, player2 goes next

    The state is defined as defined as a matrix of (4 x board_size x board_size).
        board_state[0]: current player's stones
        board_state[1]: opponent's stones
        board_state[2]: last move location
        board_state[3]: indicate which one (colour) to play
            all 1s, if player1's turn
            all 0s, if player2's turn

    `board_size`: `int` (width or height of the board)
    `n_in_a_row`: `int` (number of rocks in a row to win)
    `start_player`: `int` (either should be 1 or 2)
    """

    def __init__(self, board_size: int = 9, n_in_a_row: int = 5, start_player: int = 1):
        if board_size < n_in_a_row:
            raise Exception(
                f"Board size ({board_size}) must be >= N-in-a-row ({n_in_a_row})"
            )
        self.board_size = board_size
        self.n_in_a_row = n_in_a_row
        self.players = [1, 2]

        self.init_board(start_player)

    def init_board(self, start_player: int = 1):
        """
        `board`: `np.ndarray` (flat array)
            index = row * board_size + col
            each cell indicates player-specific stones
        """
        self.board = np.zeros(self.board_size * self.board_size, dtype=np.int8)
        self.availables = set(range(self.board_size * self.board_size))

        self.current_player = start_player
        self.last_move = -1
        self.move_counts = 0

        # For Neural Network input, used `np.float32`
        self.board_state = np.zeros(
            (4, self.board_size, self.board_size), dtype=np.float32
        )

    def reset(self, start_player: int = 1):
        self.board[:] = 0
        self.availables = set(range(self.board_size * self.board_size))

        self.current_player = start_player
        self.last_move = -1
        self.move_counts = 0

        self.board_state[:] = 0.0
        if start_player == self.players[0]:
            self.board_state[3, :, :] = 1.0

    def move_to_location(self, move: int) -> tuple[int, int]:
        """
        If given a 3 x 3 board, the board is as such:
        0 1 2
        3 4 5
        6 7 8
        If the `move` 5 is given, `location` is [1, 2]
        """
        row = move // self.board_size
        col = move % self.board_size
        return row, col

    def location_to_move(self, location: list[int]) -> int:
        if len(location) != 2:
            return -1
        row, col = location[0], location[1]
        move = row * self.board_size + col
        if move not in range(self.board_size * self.board_size):
            return -1
        return move

    def get_current_player(self) -> int:
        return self.current_player

    def play_move(self, move: int):
        # assert move in self.availables
        row, col = self.move_to_location(move)

        self.board[move] = self.current_player
        self.board_state[0, row, col] = 1.0

        if self.last_move >= 0:
            last_row, last_col = self.move_to_location(self.last_move)
            self.board_state[2, last_row, last_col] = 0.0
        self.board_state[2, row, col] = 1.0

        # Swap channels to shift perspective to the next player
        temp_state = self.board_state[0].copy()
        self.board_state[0] = self.board_state[1].copy()
        self.board_state[1] = temp_state
        self.board_state[3, :, :] = 1.0 - self.board_state[3, 0, 0]

        self.availables.remove(move)
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]

        self.last_move = move
        self.move_counts += 1

    def current_state(self) -> np.ndarray:
        return self.board_state

    def get_legal_moves(self) -> list[int]:
        return list(self.availables)

    def has_winner(self) -> tuple[bool, int]:
        """
        Returns `(has_winner: bool, won_player: int)`, based on the `self.last_move`
        Enough moves have to be preceded to decide,
            otherwise returns (False, -1)
        """
        if self.last_move == -1:
            return False, -1

        if self.move_counts < self.n_in_a_row * 2 - 1:
            return False, -1

        row, col = self.move_to_location(self.last_move)
        last_player = self.board[self.last_move]

        directions = [
            (0, 1),  # right-horizontal
            (1, 0),  # down-vertical
            (1, 1),  # down-right-diagonal
            (1, -1),  # down-left-diagonal
        ]
        b_size = self.board_size
        for dir_r, dir_c in directions:
            count = 1
            # Forward search
            r = row + dir_r
            c = col + dir_c
            while 0 <= r < b_size and 0 <= c < b_size:
                idx = r * b_size + c  # equivalent to `self.location_to_move([r, c])`
                if self.board[idx] != last_player:
                    break
                count += 1
                r += dir_r
                c += dir_c

            # Backward search
            r = row - dir_r
            c = col - dir_c
            while 0 <= r < b_size and 0 <= c < b_size:
                idx = r * b_size + c
                if self.board[idx] != last_player:
                    break
                count += 1
                r -= dir_r
                c -= dir_c

            if count >= self.n_in_a_row:
                return True, last_player

        return False, -1

    def is_game_end(self) -> tuple[bool, int]:
        """
        Returns `(is_game_ended: bool, winner: int)`
        If `winner` exists:
            `winner` is either 1 or 2,
            otherwise -1 for a draw
        """
        has_winner, winner = self.has_winner()
        if has_winner:
            return True, winner
        if not self.availables:
            return True, -1

        return False, -1

    def clone(self) -> "Board":
        cloned = Board.__new__(Board)
        cloned.board_size = self.board_size
        cloned.n_in_a_row = self.n_in_a_row
        cloned.players = self.players

        cloned.board = self.board.copy()
        cloned.availables = self.availables.copy()
        cloned.current_player = self.current_player
        cloned.last_move = self.last_move
        cloned.move_counts = self.move_counts
        cloned.board_state = self.board_state.copy()

        return cloned
