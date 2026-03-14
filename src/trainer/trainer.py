import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env.board import Board
from models.base_model import BaseModel
from agents.player import Agent
from mcts.evaluators import ModelEvaluator
from trainer.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        name: str = "alphazero",
        device: str = "cuda",
        lr: float = 1e-3,
        l2_norm: float = 1e-4,
        batch_size: int = 64,
        board_size: int = 9,
        n_in_a_row: int = 5,
        tau: float = 1.0,
        c_puct: float = 5.0,
        n_simulations: int = 400,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        log_dir: str = "runs",
        log_every: int = 20,
        eval_every: int = 100,
    ):
        self.device = device
        self.board = Board(board_size, n_in_a_row=n_in_a_row)

        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=l2_norm
        )
        self.model_evaluator = ModelEvaluator(self.model, device)
        self.agent = Agent(
            evaluator=self.model_evaluator,
            tau=tau,
            c_puct=c_puct,
            n_simulations=n_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        self.best_model = copy.deepcopy(self.model).to(device)
        self.best_model_evaluator = ModelEvaluator(self.best_model, device)
        self.best_agent = Agent(
            evaluator=self.best_model_evaluator,
            tau=tau,
            c_puct=c_puct,
            n_simulations=n_simulations,
        )
        self.best_win_rate = 0

        self.elo = 1500
        self.best_elo = 1500

        self.replay_buffer = ReplayBuffer(batch_size=batch_size, device=device)

        self.checkpoint_dir = f"{log_dir}/checkpoints_{name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log_dir = f"{log_dir}/{name}"
        self.writer = SummaryWriter(log_dir)
        self.log_every = log_every
        self.save_every = eval_every
        self.eval_every = eval_every

    def start_self_play(self, tau_threshold: int = 30):
        self.board.init_board()
        self.agent.reset()

        move_counts = 0
        states = []
        policies = []
        players = []
        winner = 0
        while True:
            if tau_threshold > move_counts:
                tau = 1.0
            else:
                tau = 0

            state = self.board.current_state()
            current_player = self.board.get_current_player()
            action, policy = self.agent.get_action(self.board, tau, True)

            states.append(state.copy())
            policies.append(policy)
            players.append(current_player)

            self.board.play_move(action)
            self.agent.mcts.update(action)
            move_counts += 1

            is_end, winner = self.board.is_game_end()
            if is_end:
                break

        # If the game's draw, then the values are zeros
        values = np.zeros(len(players))
        players = np.array(players)
        if winner != -1:
            values[players == winner] = 1
            values[players != winner] = -1

        # Store experience
        for state, policy, value in zip(states, policies, values):
            self.replay_buffer.push(state, policy, value)

        return move_counts

    def start_play(
        self, board: Board, agent: Agent, best_agent: Agent, start_player: int = 1
    ):
        board.init_board(start_player)

        agent.reset()
        agent.set_player_id(1)

        best_agent.reset()
        best_agent.set_player_id(2)

        players = {1: agent, 2: best_agent}
        while True:
            current_player_id = board.get_current_player()
            current_player = players[current_player_id]
            move, _ = current_player.get_action(board, 0, False)

            board.play_move(move)
            agent.mcts.update(move)
            best_agent.mcts.update(move)

            is_end, winner = board.is_game_end()
            if is_end:
                return winner

    def train(self):
        self.model.train()
        state, policy, value = self.replay_buffer.sample()

        pred_policy, pred_value = self.model(state)
        pred_value = pred_value.squeeze(1)

        # Policy Loss: Cross-Entropy Loss
        log_probs = F.log_softmax(pred_policy, dim=1)
        policy_loss = -(policy * log_probs).sum(dim=1).mean()

        # Value Loss: MSE Loss
        value_loss = F.mse_loss(pred_value, value)

        self.optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def fit(self, n_episodes: int, n_evals: int = 50, verbose: bool = False):
        for i in range(n_episodes):
            move_counts = self.start_self_play()

            if len(self.replay_buffer) >= self.replay_buffer.batch_size * 5:
                policy_loss, value_loss = self.train()
                if i % self.log_every == 0:
                    self.writer.add_scalar("train/loss_policy", policy_loss, i + 1)
                    self.writer.add_scalar("train/loss_value", value_loss, i + 1)
                    self.writer.add_scalar("train/game_length", move_counts, i + 1)
                    if verbose:
                        print(f" [Episode {i}]")
                        print(f"    policy_loss: {policy_loss:.3f}")
                        print(f"    value_loss: {value_loss:.3f}")

            if i % self.save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.checkpoint_dir}/model_ep{i}_{self.elo}.pth",
                )
                if verbose:
                    print(f"\t ep{i}-Model saved")

            if i % self.eval_every == 0 and i != 0:
                wins, losses, draws = self.evaluate(n_evals)
                win_rate = compute_win_rate(wins, losses, draws)

                self.writer.add_scalar("evaluation/win_rate", win_rate, i + 1)
                print(f"\t Eval | W {wins} L {losses} D {draws} |")
                print(f"\t Win Rate {win_rate:.2f} |")

                self.elo, self.best_elo = compute_ELO_rating(
                    wins, losses, draws, self.elo, self.best_elo
                )
                self.writer.add_scalar("evaluation/elo", self.elo, i + 1)
                self.writer.add_scalar("evaluation/best_elo", self.best_elo, i + 1)

                if win_rate > 0.55:
                    self.best_win_rate = win_rate
                    self.best_model.load_state_dict(self.model.state_dict())
                    torch.save(
                        self.best_model.state_dict(),
                        f"{self.checkpoint_dir}/best_model.pth",
                    )
                    self.best_model_evaluator = copy.deepcopy(self.model_evaluator)
                    self.best_agent = Agent(
                        evaluator=self.best_model_evaluator,
                        tau=0,
                        c_puct=self.agent.mcts.c_puct,
                        n_simulations=self.agent.mcts.n_simulations,
                    )

    def evaluate(self, n_evals: int = 40):
        wins, losses, draws = 0, 0, 0
        for i in range(n_evals):
            board = Board(self.board.board_size, self.board.n_in_a_row)

            if i % 2 == 0:
                start_player = 1
            else:
                start_player = 2
            winner = self.start_play(board, self.agent, self.best_agent, start_player)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1

        assert wins + losses + draws == n_evals
        return wins, losses, draws

    def _device_to(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state).to(self.device)


def compute_win_rate(wins: int, losses: int, draws: int) -> float:
    n_games = wins + losses + draws
    return wins / n_games


def compute_ELO_rating(
    wins: int,
    losses: int,
    draws: int,
    rating: float,
    opp_rating,
    k: int = 16,
) -> tuple[float, float]:
    n_games = wins + losses + draws

    # Actual Score (S)
    s = (wins + 0.5 * draws) / n_games
    opp_s = (losses + 0.5 * draws) / n_games

    # Expected Score (E)
    e = 1 / (1 + 10 ** ((opp_rating - rating) / 400))
    opp_e = 1 / (1 + 10 ** ((rating - opp_rating) / 400))

    # ELO update
    r = rating + k * (s - e)
    opp_r = opp_rating + k * (opp_s - opp_e)

    return float(r), float(opp_r)
