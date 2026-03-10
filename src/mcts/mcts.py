import numpy as np
from typing import Optional
from mcts.evaluators import PolicyValueFn
from env.board import Board


class MCTSNode:
    """
    A node of the MCTS tree.
    Each node from the tree represents a certain state of the environment,
    and each edge represents an action (move) from one state (parent node) to the other (child node)

    "parent": parent node of `MCTSNode`
    "children": child node(s) of `MCTSNode`
    "prior": prior probability from network policy head, P(s, a)
    "n_visits": visit counts, N(s, a)
    "w_sum": total accumulated value of the node, W(s, a)
    """

    __slots__ = ["parent", "children", "prior", "n_visits", "w_sum"]

    def __init__(self, parent: Optional["MCTSNode"], prior: float):
        self.parent = parent
        self.prior = prior
        self.children: dict[int, "MCTSNode"] = {}
        self.n_visits: int = 0
        self.w_sum: float = 0.0

    @property
    def Q(self) -> float:
        if self.n_visits > 0:
            return self.w_sum / self.n_visits
        else:
            return 0.0

    def is_leaf(self) -> bool:
        if len(self.children) == 0:
            return True
        else:
            return False

    def is_root(self) -> bool:
        if self.parent is None:
            return True
        else:
            return False

    def expand(self, action_prior_pairs: dict[int, float]):
        """
        Expands tree with `action_prior_pairs`
        consists of a dictionary of `action`s and its corresponding `prior`s
        """
        for action, prior in action_prior_pairs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def select(self, c_puct: float) -> tuple[int, "MCTSNode"]:
        """
        Select a child node (action) with the highest UCB score.
            UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        Arg:
            `c_puct`: float ($c_1$ or $c_{puct}$)
        Returns:
            (`best_action`: int, `best_child`: MCTSNode)
        """
        sqrt_parent = np.sqrt(self.n_visits + 1e-8)
        best_ucb = -float("inf")

        best = []
        for action, child in self.children.items():
            ucb = c_puct * child.prior * sqrt_parent / (1 + child.n_visits)
            ucb = -child.Q + ucb

            if ucb > best_ucb:
                best_ucb = ucb
                best = [(action, child)]
            elif abs(ucb - best_ucb) <= 1e-12:
                best.append((action, child))

        idx = np.random.randint(len(best))
        return best[idx]

    def backup(self, value: float):
        node = self
        v = value
        while node is not None:
            node.n_visits += 1
            node.w_sum += v
            # Since players alternate turns, values are negated to switch perspectives
            v = -v
            node = node.parent


class MCTS:
    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        c_puct: float = 5.0,
        n_simulations: int = 500,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root = MCTSNode(parent=None, prior=1.0)

    def _terminal_value(self, board: Board, winner: int) -> float:
        """
        Return a value from current player's perspective at a terminal state.
        After the winning move (action), `_terminal_value()` is called,
        and current player is always a loser.
        """
        if winner == -1:
            return 0.0
        return -1.0

    def _simulate(self, board: Board):
        """
        Runs a full simulation on a deep copied `board`.
        Select, expand, evaluate, and backup.

        Evaluation could be rollout (playout) or other functions.
        """
        node = self.root

        # Select
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            board.play_move(action)

        is_end, winner = board.is_game_end()
        if is_end:
            node.backup(self._terminal_value(board, winner))
            return

        # Evaluate and expand
        action_prior_pairs, value = self.policy_value_fn(board)
        node.expand(action_prior_pairs)

        # Backup
        node.backup(value)

    def _add_dirichlet_noise(
        self, action_prior_pairs: dict[int, float]
    ) -> dict[int, float]:
        actions = list(action_prior_pairs.keys())

        if not actions:
            return action_prior_pairs

        noises = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        noised_priors = {
            a: (1 - eps) * action_prior_pairs[a] + eps * n
            for a, n in zip(actions, noises)
        }

        total_priors = sum(noised_priors.values())
        if total_priors > 0:
            noised_priors = {a: p / total_priors for a, p in noised_priors.items()}

        return noised_priors

    def reset(self):
        self.root = MCTSNode(parent=None, prior=1.0)

    def update(self, action: int):
        """
        Moves the tree forward by switching its root to the child node
        where the latest move is executed

        Arg:
            `action`: `int` (latest move made)
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.reset()

    def search(
        self, board: Board, tau: float = 0.0, add_noise: bool = False
    ) -> tuple[int, np.ndarray]:
        """
        Runs MCTS simulations and returns an action and its policy (probability distribution).

        Args:
            `board`: `Board`
            `tau`: `float` = 1.0 (temperature argument which controls action selection)
                if `tau` approx to 0: greedy selection
                elif `tau` approx to 1: proportional selection based on the visit counts
                elif `tau` approx to inf: uniform
            `add_noise`: `bool` = False (Dirichlet noise added to priors of nodes)

        Returns:
            `action`: `int` (chosen action)
            `policy`: `np.ndarray` (policy vector with a size of $board_size^2$)
        """
        if self.root.is_leaf():
            action_prior_pairs, _ = self.policy_value_fn(board)
            if add_noise:
                action_prior_pairs = self._add_dirichlet_noise(action_prior_pairs)
            self.root.expand(action_prior_pairs)

        # Simulations on a cloned board, then update the tree
        for _ in range(self.n_simulations):
            self._simulate(board.clone())

        b_size_sq = board.board_size * board.board_size
        # Make a policy vector (probability distribution)
        visits = np.zeros(b_size_sq, dtype=np.float32)
        for action, child in self.root.children.items():
            visits[action] = child.n_visits

        v = visits.astype(np.float32)
        if tau <= 1e-3:  # temperature near zero or equal to zero
            action = int(np.argmax(v))
            policy = np.zeros_like(v)
            policy[action] = 1.0
            return action, policy

        # pi(a) \leftarrow N(a)^{1 / tau}
        v = v ** (1.0 / tau)
        v_sum = v.sum()
        if v_sum == 0:
            legal = board.get_legal_moves()
            policy = np.zeros_like(v)
            policy[legal] = 1.0 / len(legal)
            action = np.random.choice(legal)
            return action, policy

        policy = (v / v_sum).astype(np.float32)
        action = int(np.random.choice(np.arange(len(v)), p=policy))

        return action, policy
