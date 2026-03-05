import argparse
import time
from env.board import Board
from env.gomoku import Gomoku
from agents.player import RandomPlayer
from agents.mcts_agent import MCTSAgent


def run_benchmark(
    n_simulations: int,
    n_games: int,
    board_size: int = 9,
    n_in_a_row: int = 5,
) -> dict:
    """
    Pit MCTSAgent vs RandomPlayer over n_games.
    Alternates who goes first each game.
    """
    wins = losses = draws = 0
    total_time = 0.0
    total_moves = 0

    for i in range(n_games):
        board = Board(board_size, n_in_a_row)
        game = Gomoku(board)

        agent = MCTSAgent(n_simulations=n_simulations, seed=i)
        random = RandomPlayer(seed=i * 100)

        # Alternate who goes first
        mcts_id = 1 if i % 2 == 0 else 2
        random_id = 2 if i % 2 == 0 else 1
        agent.set_player_id(mcts_id)
        random.set_player_id(random_id)

        t0 = time.time()
        if i % 2 == 0:
            winner = game.start_play(agent, random, start_player=1)
        else:
            winner = game.start_play(agent, random, start_player=2)
        elapsed = time.time() - t0

        total_time += elapsed
        total_moves += board.move_counts

        if winner == mcts_id:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1

        agent.reset()

        result_str = (
            "MCTS" if winner == mcts_id else "Draw" if winner == -1 else "Random"
        )
        print(
            f"  Game {i+1:3d}/{n_games} | "
            f"W={wins} L={losses} D={draws} | "
            f"winner={result_str}"
        )

    return {
        "n_simulations": n_simulations,
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "avg_game_time": total_time / n_games,
        "avg_move_time": total_time / total_moves if total_moves > 0 else 0,
    }


def print_results(r: dict) -> None:
    n = r["n_games"]
    print(f"\n{'='*45}")
    print(f"  Pure MCTS ({r['n_simulations']} sims) vs Random — {n} games")
    print(f"{'='*45}")
    print(f"  Wins     : {r['wins']:3d}  ({r['wins']/n:.1%})")
    print(f"  Losses   : {r['losses']:3d}  ({r['losses']/n:.1%})")
    print(f"  Draws    : {r['draws']:3d}  ({r['draws']/n:.1%})")
    print(f"  Win rate : {r['win_rate']:.1%}")
    print(f"  Avg game : {r['avg_game_time']:.2f}s")
    print(f"  Avg move : {r['avg_move_time']*1000:.1f}ms")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, nargs="+", default=[400])
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--n_in_a_row", type=int, default=5)
    args = parser.parse_args()

    for sims in args.sims:
        print(f"\nRunning: {sims} simulations × {args.games} games...")
        results = run_benchmark(sims, args.games, args.board_size, args.n_in_a_row)
        print_results(results)
