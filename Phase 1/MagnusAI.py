import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Dropout, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import chess
import random
from collections import deque
import os
import pickle
import time

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}            # move -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_expanded(self):
        return len(self.children) > 0


class MagnusChessAI:
    def __init__(self, model_path: str = "magnus_model.keras"):
        self.model_path = model_path
        self.model = None
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._initialize_move_mapping()
        self.num_unique_moves = len(self.move_to_idx)
        self.training_data = deque(maxlen=20000)  # store recent self-play examples

        # MCTS hyperparameters
        self.C_PUCT = 1.5

    def _initialize_move_mapping(self):
        """Create mapping between all reasonable chess move UCIs and indices.
        This generates moves by iterating from-square to to-square and creating
        promotion variants. The resulting mapping is used for policy output ordering.
        """
        moves = set()
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue
                # basic move
                try:
                    move = chess.Move(from_sq, to_sq)
                    moves.add(move.uci())
                except Exception:
                    pass
                # promotions (white rank 7 -> 8, black 2 -> 1)
                from_rank = chess.square_rank(from_sq)
                to_rank = chess.square_rank(to_sq)
                if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                    for promo in ['q', 'r', 'b', 'n']:
                        try:
                            pm = chess.Move(from_sq, to_sq, promotion=chess.PIECE_SYMBOLS.index(promo))
                        except Exception:
                            # fallback: build UCI string manually
                            pm_uci = chess.square_name(from_sq) + chess.square_name(to_sq) + promo
                            moves.add(pm_uci)
                            continue
                        moves.add(pm.uci())
        moves_list = sorted(list(moves))
        self.move_to_idx = {m: i for i, m in enumerate(moves_list)}
        self.idx_to_move = {i: m for m, i in self.move_to_idx.items()}
        print(f"Initialized move mapping with {len(moves_list)} moves")

    def fen_to_tensor(self, fen: str) -> np.ndarray:
        """Convert FEN to tensor with shape (8,8,12).
        Use chess.square_rank and square_file for correct orientation.
        """
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        board_tensor = np.zeros((8, 8, 12), dtype=np.float32)
        board = chess.Board(fen)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = chess.square_rank(square)  # 0..7 (rank 1 -> 0)
                col = chess.square_file(square)  # 0..7 (file a -> 0)
                board_tensor[row, col, piece_map[piece.symbol()]] = 1.0
        # Note: we intentionally do not add extra feature planes here; keep input consistent
        return board_tensor

    def build_model(self) -> Model:
        """Build dual-head network (policy + value).
        Policy output is a softmax over all move UCIs in move_to_idx.
        Value output is tanh in [-1,1] representing advantage for the player to move.
        """
        board_input = Input(shape=(8, 8, 12), name='board_input')

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        # simple residual
        residual = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])

        # Policy head
        p = Conv2D(32, (1, 1), activation='relu')(x)
        p = BatchNormalization()(p)
        p = Flatten()(p)
        p = Dense(256, activation='relu')(p)
        p = Dense(self.num_unique_moves, activation='softmax', name='policy_output')(p)

        # Value head
        v = Conv2D(32, (1, 1), activation='relu')(x)
        v = BatchNormalization()(v)
        v = Flatten()(v)
        v = Dense(128, activation='relu')(v)
        v = Dropout(0.3)(v)
        v = Dense(1, activation='tanh', name='value_output')(v)

        model = Model(inputs=board_input, outputs=[p, v])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
            loss_weights={'policy_output': 1.0, 'value_output': 1.0},
            metrics={'policy_output': 'accuracy', 'value_output': 'mse'}
        )
        return model

    def save_model(self, model_path: str = None):
        if model_path is None:
            model_path = self.model_path
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        self.model.save(model_path)
        mapping_path = model_path.replace('.keras', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump({'move_to_idx': self.move_to_idx, 'idx_to_move': self.idx_to_move}, f)
        print(f"Saved model to {model_path} and mapping to {mapping_path}")

    def load_model(self, model_path: str = None):
        if model_path is None:
            model_path = self.model_path
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return False
        self.model = tf.keras.models.load_model(model_path)
        mapping_path = model_path.replace('.keras', '_mapping.pkl')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                data = pickle.load(f)
                self.move_to_idx = data['move_to_idx']
                self.idx_to_move = data['idx_to_move']
                self.num_unique_moves = len(self.move_to_idx)
        print(f"Loaded model from {model_path}")
        return True

    def _predict(self, board: chess.Board):
        """Return (policy_probs, value) for a board. Policy is over move_to_idx ordering.
        Value is in [-1,1] from perspective of the player to move.
        """
        tensor = self.fen_to_tensor(board.fen())
        tensor = np.expand_dims(tensor, axis=0)
        policy, value = self.model.predict(tensor, verbose=0)
        return policy[0], float(value[0][0])

    def random_move(self, board: chess.Board):
        return random.choice(list(board.legal_moves))

    def play_move(self, board: chess.Board, simulations: int = 100, temperature: float = 1.0):
        """Public API: run MCTS and return a selected move (chess.Move).
        If model not trained enough, fall back to random moves.
        """
        if self.model is None or len(self.training_data) < 50:
            return self.random_move(board)

        root = self.run_mcts(board, simulations)
        # choose move by most visits (temperature handling)
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=float)
        if temperature == 0:
            best_idx = int(np.argmax(visits))
            return moves[best_idx]
        probs = visits ** (1.0 / max(1e-8, temperature))
        probs = probs / probs.sum()
        return random.choices(moves, weights=probs, k=1)[0]

    def run_mcts(self, board: chess.Board, simulations: int = 100):
        root = MCTSNode(board)
        # Expand root with network priors
        policy, _ = self._predict(board)
        legal = list(board.legal_moves)
        for mv in legal:
            mv_uci = mv.uci()
            idx = self.move_to_idx.get(mv_uci, None)
            prior = float(policy[idx]) if idx is not None and idx < len(policy) else 1e-6
            child_board = board.copy()
            child_board.push(mv)
            child = MCTSNode(child_board, parent=root)
            child.prior = prior
            root.children[mv] = child

        for _ in range(simulations):
            node = root
            path = [node]

            # Selection
            while node.is_expanded() and not node.board.is_game_over():
                node = self._select_child(node)
                path.append(node)

            # Expansion & Evaluation
            value = self._evaluate_node(node)

            # Backpropagation (flip sign each level)
            self._backpropagate(path, value)

        return root

    def _select_child(self, node: MCTSNode):
        total_N = sum(child.visits for child in node.children.values())
        best_score = -float('inf')
        best_child = None
        for move, child in node.children.items():
            Q = child.value
            U = self.C_PUCT * child.prior * np.sqrt(max(1.0, total_N)) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _evaluate_node(self, node: MCTSNode):
        # Terminal node
        if node.board.is_game_over():
            # return value in [-1,1] from perspective of player to move at this node
            result = node.board.result(claim_draw=True)
            if result == '1-0':
                return 1.0 if node.board.turn == chess.WHITE else -1.0
            elif result == '0-1':
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            else:
                return 0.0

        # Non-terminal: use network
        policy, value = self._predict(node.board)
        # Expand if not expanded
        if not node.is_expanded():
            for mv in node.board.legal_moves:
                mv_uci = mv.uci()
                idx = self.move_to_idx.get(mv_uci, None)
                prior = float(policy[idx]) if idx is not None and idx < len(policy) else 1e-8
                child_board = node.board.copy()
                child_board.push(mv)
                child = MCTSNode(child_board, parent=node)
                child.prior = prior
                node.children[mv] = child
        # value is already from perspective of player to move
        return value

    def _backpropagate(self, path, value):
        # Value is from perspective of the leaf node's player to move.
        # We propagate it up the tree, flipping sign at each step so that
        # value is always interpreted as "from the perspective of node.board.turn".
        v = value
        for node in reversed(path):
            node.visits += 1
            node.value_sum += v
            v = -v

    def _get_game_result(self, board: chess.Board):
        # Return final game result as +1 (white win), -1 (black win), 0 draw
        res = board.result(claim_draw=True)
        if res == '1-0':
            return 1.0
        elif res == '0-1':
            return -1.0
        else:
            return 0.0

    # ---------------- Self-play & training ----------------
    def collect_self_play_data(self, num_games=20, simulations_per_move=200):
        """Generate self-play games using current policy+MCTS and store training examples.
        Each example is (board_state, policy_target (visit counts normalized), value_target (final result from perspective of player to move)).
        """
        print(f"Starting self-play for {num_games} games...")
        for gi in range(num_games):
            board = chess.Board()
            history = []  # list of (board_copy, visit_counts_array)
            while not board.is_game_over():
                root = self.run_mcts(board, simulations=simulations_per_move)
                # build visit count array
                visit_counts = np.zeros(self.num_unique_moves, dtype=float)
                for mv, child in root.children.items():
                    idx = self.move_to_idx.get(mv.uci(), None)
                    if idx is not None:
                        visit_counts[idx] = child.visits
                total = visit_counts.sum()
                if total > 0:
                    visit_probs = visit_counts / total
                else:
                    visit_probs = np.ones_like(visit_counts) / len(visit_counts)
                history.append((board.copy(), visit_probs))
                # choose move (most visited)
                best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
                board.push(best_move)
            # final result
            result = self._get_game_result(board)
            # store examples with value target from perspective of player to move
            for bstate, move_probs in history:
                value_target = result if bstate.turn == chess.WHITE else -result
                self.training_data.append((bstate, move_probs, value_target))
            print(f"Game {gi+1}/{num_games} finished: {board.result()} -- collected {len(history)} positions")
        print(f"Total training samples: {len(self.training_data)}")

    def train(self, epochs=5, batch_size=64):
        if not self.training_data:
            print("No training data. Run self-play first.")
            return None
        # prepare arrays
        boards = np.array([self.fen_to_tensor(b.fen()) for b, _, _ in self.training_data])
        policies = np.array([p for _, p, _ in self.training_data])
        values = np.array([v for _, _, v in self.training_data])

        print(f"Training on {len(boards)} samples")
        history = self.model.fit(
            boards,
            {'policy_output': policies, 'value_output': values},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history

    def evaluate_ai(self, num_games=5, simulations=50):
        wins = losses = draws = 0
        for gi in range(num_games):
            board = chess.Board()
            ai_color = chess.WHITE if gi % 2 == 0 else chess.BLACK
            while not board.is_game_over():
                if board.turn == ai_color:
                    mv = self.play_move(board, simulations=simulations, temperature=0)
                else:
                    mv = self.random_move(board)
                board.push(mv)
            res = board.result()
            print(f"Eval game {gi+1}: {res}")
            if res == '1-0':    
                if ai_color == chess.WHITE:
                    wins += 1
                else:
                    losses += 1
            elif res == '0-1':
                if ai_color == chess.BLACK:
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1
        print(f"Eval results: {wins} wins, {losses} losses, {draws} draws")
        return wins, losses, draws


# Example training pipeline (simple)
if __name__ == '__main__':
    ai = MagnusChessAI()
    # Build a new model if not loading
    if not ai.load_model():
        ai.model = ai.build_model()
        print("Built new model")

    # Bootstrap with random play
    ai.random_play = lambda n: [None]  # keep a placeholder; user can implement if desired
    # Do several self-play / train iterations
    for iteration in range(10):
        print(f"=== Iteration {iteration+1} ===")
        ai.collect_self_play_data(num_games=15, simulations_per_move=30)
        ai.train(epochs=5, batch_size=64)
        ai.save_model()
    print("Done")
