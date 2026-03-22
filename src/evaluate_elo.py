"""
Evaluate AttnAlphaZero checkpoints against the fixed AlphaZero best model
and plot relative ELO over training episodes.

Usage:
    python src/evaluate_elo.py
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))

from models.attn_model import AttnPolicyValue
from models.policy_value_model import PolicyValueModel
from mcts.evaluators import ModelEvaluator
from agents.player import Agent
from env.board import Board

BASELINE_PATH = "runs/testing/alphazero/best_model.pth"
CHECKPOINT_DIR = "runs/AttnAlphaZero/checkpoints_AttnAlphaZero"
OUTPUT_PATH = "runs/elo_evaluation.png"
BASELINE_ELO = 1500.0

N_EVALS = 20  # games per checkpoint
N_SIMULATIONS = 400
C_PUCT = 3.0
BOARD_SIZE = 9
N_IN_A_ROW = 5

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_baseline(path: str) -> Agent:
    model = PolicyValueModel(BOARD_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    evaluator = ModelEvaluator(model, DEVICE)
    return Agent(evaluator, c_puct=C_PUCT, n_simulations=N_SIMULATIONS)


def _make_attn_agent(path: str) -> Agent:
    model = AttnPolicyValue(
        board_size=BOARD_SIZE, n_in_a_row=N_IN_A_ROW, n_blocks=1
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    evaluator = ModelEvaluator(model, DEVICE)
    return Agent(evaluator, c_puct=C_PUCT, n_simulations=N_SIMULATIONS)


def _start_play(
    board: Board, agent1: Agent, agent2: Agent, start_player: int = 1
) -> int:
    board.init_board(start_player)
    agent1.reset()
    agent1.set_player_id(1)
    agent2.reset()
    agent2.set_player_id(2)
    players = {1: agent1, 2: agent2}

    move_counts = 0
    while True:
        tau = 0.1 if move_counts <= 3 else 0
        current_id = board.get_current_player()
        move, _ = players[current_id].get_action(board, tau, False)
        board.play_move(move)
        agent1.mcts.update(move)
        agent2.mcts.update(move)
        move_counts += 1
        is_end, winner = board.is_game_end()
        if is_end:
            return winner


def run_evaluation(
    attn_agent: Agent, baseline: Agent, n_evals: int
) -> tuple[int, int, int]:
    wins, losses, draws = 0, 0, 0
    for i in range(n_evals):
        board = Board(BOARD_SIZE, N_IN_A_ROW)
        if i % 2 == 0:
            winner = _start_play(board, attn_agent, baseline, start_player=1)
            attn_player_id = 1
        else:
            winner = _start_play(board, baseline, attn_agent, start_player=1)
            attn_player_id = 2

        if winner == attn_player_id:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1

    return wins, losses, draws


def win_rate_to_relative_elo(
    win_rate: float, baseline_elo: float = BASELINE_ELO
) -> float:
    wr = np.clip(win_rate, 1e-5, 1 - 1e-5)
    return baseline_elo - 400 * np.log10(1 / wr - 1)


def parse_episode(filename: str) -> int:
    m = re.search(r"model_ep(\d+)_", filename)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading baseline from {BASELINE_PATH}")
    baseline = _make_baseline(BASELINE_PATH)

    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_ep")],
        key=parse_episode,
    )
    print(f"Found {len(checkpoints)} checkpoints")

    episodes, elos = [], []

    for i, ckpt in enumerate(checkpoints):
        ep = parse_episode(ckpt)
        path = os.path.join(CHECKPOINT_DIR, ckpt)

        attn_agent = _make_attn_agent(path)
        wins, losses, draws = run_evaluation(attn_agent, baseline, N_EVALS)
        wr = wins / N_EVALS
        elo = win_rate_to_relative_elo(wr)

        episodes.append(ep)
        elos.append(elo)
        print(
            f"  [{i+1}/{len(checkpoints)}] ep{ep:5d}  W{wins} L{losses} D{draws}  elo={elo:.0f}"
        )

    episodes = np.array(episodes)
    elos = np.array(elos)

    fig, ax = plt.subplots()
    ax.plot(
        episodes,
        elos,
        color="orange",
        marker="o",
        markersize=3,
        label="Attention AlphaZero",
    )
    ax.axhline(
        BASELINE_ELO,
        color="blue",
        linestyle="--",
        label=f"AlphaZero best model (ELO {BASELINE_ELO:.0f})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Relative ELO")
    ax.set_title("ELO Evaluation: Attention AlphaZero vs AlphaZero")
    ax.legend()
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print(f"\nPlot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
