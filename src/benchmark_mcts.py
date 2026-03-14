import argparse
import time
from env.board import Board
from env.gomoku import Gomoku
from agents.player import RandomPlayer, Agent
from mcts.evaluators import RandomEvaluator


def compute_ELO_rating(
    game_result: dict[str, float | int],
    rating,
    opp_rating,
    k: int = 16,
):
    wins = game_result["wins"]
    draws = game_result["draws"]
    n_games = game_result["n_games"]
    # Actual Score (S)
    s = (wins + 0.5 * draws) / n_games
    # Expected Score (E)
    e = 1 / (1 + 10 ** ((opp_rating - rating) / 400))
    # ELO update
    r = rating + k * (s - e)
    return float(r)


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

    game_result = {}
    rating = 1500
    opp_rating = 1500
    for i in range(n_games):
        board = Board(board_size, n_in_a_row)
        game = Gomoku(board)

        mcts_evaluator = RandomEvaluator(i)
        agent = Agent(mcts_evaluator, n_simulations=n_simulations)
        random = RandomPlayer(seed=i * 100)

        t0 = time.time()
        if i % 2 == 0:
            winner = game.start_play_with_random(agent, random, start_player=1)
        else:
            winner = game.start_play_with_random(agent, random, start_player=2)
        elapsed = time.time() - t0

        total_time += elapsed
        total_moves += board.move_counts

        if winner == 1:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1

        agent.reset()

        result_str = "MCTS" if winner == 1 else "Draw" if winner == -1 else "Random"

        game_result["n_games"] = i + 1
        game_result["wins"] = wins
        game_result["draws"] = draws
        new_rating = compute_ELO_rating(game_result, rating, opp_rating)
        rating = float(new_rating)
        # print(
        #     f"  Game {i+1:3d}/{n_games} | "
        #     f"W={wins} L={losses} D={draws} | "
        #     f"winner={result_str}"
        #     f"\n\tELO Rating={rating:.2f}\t{winner}"
        # )

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


def run_benchmark2(
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

    game_result = {}
    rating = 1500
    opp_rating = 1500
    for i in range(n_games):
        board = Board(board_size, n_in_a_row)
        game = Gomoku(board)

        # agent = MCTSAgent(n_simulations=n_simulations, seed=i)
        mcts_evaluator = RandomEvaluator(i)
        agent = Agent(mcts_evaluator, n_simulations=n_simulations)
        mcts_evaluator2 = RandomEvaluator(i * 100)
        agent2 = Agent(mcts_evaluator2, n_simulations=50)

        t0 = time.time()
        if i % 2 == 0:
            winner = game.start_play(agent, agent2, start_player=1)
        else:
            winner = game.start_play(agent, agent2, start_player=2)
        elapsed = time.time() - t0

        total_time += elapsed
        total_moves += board.move_counts

        if winner == 1:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1

        agent.reset()
        agent2.reset()

        result_str = "MCTS" if winner == 1 else "Draw" if winner == -1 else "MCTS200"

        game_result["n_games"] = i + 1
        game_result["wins"] = wins
        game_result["draws"] = draws
        new_rating = compute_ELO_rating(game_result, rating, opp_rating)
        rating = float(new_rating)
        # print(
        #     f"  Game {i+1:3d}/{n_games} | "
        #     f"W={wins} L={losses} D={draws} | "
        #     f"winner={result_str}"
        #     f"\n\tELO Rating={rating:.2f}"
        # )

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


def print_results(r: dict, opp="MCTS (100)") -> None:
    n = r["n_games"]
    print(f"\n{'='*45}")
    print(f"  Pure MCTS ({r['n_simulations']} sims) vs {opp} — {n} games")
    print(f"{'='*45}")
    print(f"  Wins     : {r['wins']:3d}  ({r['wins']/n:.1%})")
    print(f"  Losses   : {r['losses']:3d}  ({r['losses']/n:.1%})")
    print(f"  Draws    : {r['draws']:3d}  ({r['draws']/n:.1%})")
    print(f"  Win rate : {r['win_rate']:.1%}")
    print(f"  Avg game : {r['avg_game_time']:.2f}s")
    print(f"  Avg move : {r['avg_move_time']*1000:.1f}ms")
    print(f"{'='*45}\n")


def mcts_vs_random():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, nargs="+", default=[100])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--n_in_a_row", type=int, default=5)
    args = parser.parse_args()

    for sims in args.sims:
        print(f"\nRunning: {sims} simulations X {args.games} games...")
        results = run_benchmark(sims, args.games, args.board_size, args.n_in_a_row)
        print_results(results, "Random")


def mcts_vs_mcts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--n_in_a_row", type=int, default=5)
    args = parser.parse_args()

    for sims in args.sims:
        print(f"\nRunning: {sims} simulations X {args.games} games...")
        results = run_benchmark2(sims, args.games, args.board_size, args.n_in_a_row)
        print_results(results, f"MCTS (50)")


if __name__ == "__main__":
    # mcts_vs_random()
    mcts_vs_mcts()
