import sys
import pygame
from typing import Optional

from env.board import Board
from env.gomoku import Gomoku
from env.player import HumanPlayer

BOARD_BG = (205, 155, 75)
GRID_LINE = (70, 40, 20)
HIGHLIGHT = (255, 180, 50)
BLACK_STONE = (30, 30, 30)
WHITE_STONE = (245, 245, 245)
UI_BG = (60, 60, 60)
UI_TEXT = (220, 220, 220)
UI_ACCENT = (255, 200, 60)
UI_DIVIDER = (80, 80, 80)
WIN_OVERLAY = (0, 0, 0, 160)

SIDEBAR_W = 220
MARGIN = 40

PLAYER_SYMBOL = {1: "X", 2: "O"}


class GomokuGUI:
    """
    Pygame GUI for Gomoku — Human vs Human.

    Args:
        board_size (int): number of lines on each side
        n_in_a_row (int): consecutive stones needed to win
        cell_size (int): pixels between grid lines
        start_player (int): which player moves first (1 or 2)
    """

    def __init__(
        self,
        board_size: int = 9,
        n_in_a_row: int = 5,
        cell_size: int = 60,
        start_player: int = 1,
    ):
        self.board_size = board_size
        self.n_in_a_row = n_in_a_row
        self.cell_size = cell_size
        self.start_player = start_player

        self.board_px = (board_size - 1) * cell_size
        self.window_w = self.board_px + 2 * MARGIN + SIDEBAR_W
        self.window_h = self.board_px + 2 * MARGIN
        self.stone_r = int(cell_size * 0.3)

        self.board = Board(board_size, n_in_a_row, start_player)
        self.game = Gomoku(self.board)

        self.player1 = HumanPlayer(player_name="Player 1")
        self.player2 = HumanPlayer(player_name="Player 2")
        self.game.assign_players(self.player1, self.player2)
        self.players = {1: self.player1, 2: self.player2}

        self.game_over = False
        self.winner = -1
        self.status_msg = self._turn_message()

        pygame.init()
        pygame.display.set_caption("Gomoku")
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.font_lg = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_sm = pygame.font.SysFont("Arial", 16)
        self.clock = pygame.time.Clock()

    def run(self):
        while True:
            self._handle_events()
            self._draw()
            self.clock.tick(60)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._restart()
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                move = self._mouse_coord_to_move(event.pos)
                if move is not None and move in self.board.availables:
                    self._apply_move(move)

    def _mouse_coord_to_move(self, pos: tuple[int, int]) -> Optional[int]:
        x, y = pos
        col = round((x - MARGIN) / self.cell_size)
        row = round((y - MARGIN) / self.cell_size)
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row * self.board_size + col
        return None

    def _apply_move(self, move: int):
        self.board.play_move(move)
        is_ended, winner = self.board.is_game_end()
        if is_ended:
            self.game_over = True
            self.winner = winner
            if winner == -1:
                self.status_msg = "Draw!"
            else:
                self.status_msg = f"{self.players[winner].player_name} wins!"
        else:
            self.status_msg = self._turn_message()

    def _restart(self):
        self.board.init_board(self.start_player)
        self.game_over = False
        self.winner = -1
        self.status_msg = self._turn_message()

    def _draw(self):
        self.screen.fill(BOARD_BG)
        self._draw_grid()
        self._draw_star_points()
        self._draw_stones()
        self._draw_last_move_marker()
        self._draw_sidebar()
        if self.game_over:
            self._draw_win_overlay()
        pygame.display.flip()

    def _draw_grid(self):
        for i in range(self.board_size):
            pygame.draw.line(
                self.screen,
                GRID_LINE,
                (MARGIN, MARGIN + i * self.cell_size),
                (MARGIN + self.board_px, MARGIN + i * self.cell_size),
                1,
            )
            pygame.draw.line(
                self.screen,
                GRID_LINE,
                (MARGIN + i * self.cell_size, MARGIN),
                (MARGIN + i * self.cell_size, MARGIN + self.board_px),
                1,
            )

    def _draw_star_points(self):
        star_map = {
            9: [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)],
            13: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)],
            15: [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)],
        }
        for r, c in star_map.get(self.board_size, []):
            cx = MARGIN + c * self.cell_size
            cy = MARGIN + r * self.cell_size
            pygame.draw.circle(self.screen, GRID_LINE, (cx, cy), 4)

    def _draw_stones(self):
        board_2d = self.board.board.reshape(self.board_size, self.board_size)
        for row in range(self.board_size):
            for col in range(self.board_size):
                player = board_2d[row, col]
                if player != 0:
                    cx = MARGIN + col * self.cell_size
                    cy = MARGIN + row * self.cell_size
                    self._draw_stone(cx, cy, player)

    def _draw_stone(self, cx: int, cy: int, player: int):
        colour = BLACK_STONE if player == 1 else WHITE_STONE
        pygame.draw.circle(self.screen, colour, (cx, cy), self.stone_r)

    def _draw_last_move_marker(self):
        if self.board.last_move < 0:
            return
        row, col = self.board.move_to_location(self.board.last_move)
        cx = MARGIN + col * self.cell_size
        cy = MARGIN + row * self.cell_size
        pygame.draw.circle(self.screen, HIGHLIGHT, (cx, cy), self.stone_r // 4)

    def _draw_sidebar(self):
        sx = self.board_px + 2 * MARGIN
        pygame.draw.rect(self.screen, UI_BG, (sx, 0, SIDEBAR_W, self.window_h))

        def row(text, y, font, colour, gap=25):
            self._text(text, sx + 10, y, font, colour)
            return y + gap

        def divider(y, gap=15):
            pygame.draw.line(
                self.screen, UI_DIVIDER, (sx + 5, y), (sx + SIDEBAR_W - 5, y), 1
            )
            return y + gap

        y = 20
        y = row("GOMOKU", y, self.font_lg, UI_ACCENT, gap=40)
        y = row(
            f"{self.player1.player_name} vs {self.player2.player_name}",
            y,
            self.font_sm,
            UI_TEXT,
            gap=30,
        )
        y = divider(y)
        y = row("Status:", y, self.font_sm, UI_ACCENT)
        y = row(self.status_msg, y, self.font_sm, UI_TEXT, gap=30)
        y = row(f"Move: {self.board.move_counts}", y, self.font_sm, UI_TEXT, gap=35)
        y = divider(y, gap=20)

        for player_id in (1, 2):
            colour = BLACK_STONE if player_id == 1 else WHITE_STONE
            name = self.players[player_id].player_name
            symbol = PLAYER_SYMBOL[player_id]
            pygame.draw.circle(self.screen, colour, (sx + 22, y + 10), 10)
            self._text(f"{name} ({symbol})", sx + 40, y + 2, self.font_sm, UI_TEXT)
            y += 30

        y = divider(y + 10, gap=10)
        y = row("Controls:", y, self.font_sm, UI_ACCENT)
        y = row("Click - place", y, self.font_sm, UI_TEXT)
        y = row("R - restart", y, self.font_sm, UI_TEXT)
        y = row("Q - quit", y, self.font_sm, UI_TEXT)

    def _draw_win_overlay(self):
        overlay = pygame.Surface(
            (self.board_px + 2 * MARGIN, self.window_h), pygame.SRCALPHA
        )
        overlay.fill(WIN_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        cx = (self.board_px + 2 * MARGIN) // 2
        cy = self.window_h // 2
        t1 = self.font_lg.render(self.status_msg, True, UI_ACCENT)
        t2 = self.font_sm.render("R to restart  |  Q to quit", True, UI_TEXT)
        self.screen.blit(t1, t1.get_rect(center=(cx, cy - 20)))
        self.screen.blit(t2, t2.get_rect(center=(cx, cy + 20)))

    def _text(self, text: str, x: int, y: int, font, colour):
        self.screen.blit(font.render(text, True, colour), (x, y))

    def _turn_message(self) -> str:
        """Uses PLAYER_SYMBOL — no inline ternary needed."""
        p = self.board.get_current_player()
        return f"{self.players[p].player_name} ({PLAYER_SYMBOL[p]})"


def play_pygame(
    board_size: int = 9,
    n_in_a_row: int = 5,
    cell_size: int = 60,
    start_player: int = 1,
):
    GomokuGUI(board_size, n_in_a_row, cell_size, start_player).run()
