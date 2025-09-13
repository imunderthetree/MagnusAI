# MagnusAI – Phase 1 🏆

*Building a chess-playing AI from scratch using deep learning and Monte Carlo Tree Search*

This is the first phase of the MagnusAI project: an attempt to train a chess-playing AI from scratch using **deep learning** and **Monte Carlo Tree Search (MCTS)**. The code in this phase builds the foundation of the system with a self-play training loop where the AI improves by playing against itself.

An interactive chess interface is already developed, but it will only be released in the final phase once the AI is strong enough to make it meaningful.

## 🎮 Training Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Chess      │───▶│ MCTS Search  │───▶│ Move Select │───▶│ Game Outcome │
│  Position   │    │ (200 sims)   │    │ (by visits) │    │  (+1/0/-1)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       ▲                                                           │
       │           ┌──────────────┐    ┌─────────────┐            │
       └───────────│ Updated NN   │◀───│ Training    │◀───────────┘
                   │ Predictions  │    │ Batch       │
                   └──────────────┘    └─────────────┘
```

## 🚀 Features in Phase 1

### 🧠 Neural Network Architecture
- **Dual-head design**: Policy head (move probabilities) + Value head (position evaluation)
- **Convolutional layers**: Feature extraction from 8×8×12 board representation
- **Residual connection**: Improved gradient flow and training stability
- **Output dimensions**: 
  - Policy: ~4,096 possible moves (all UCI move strings)
  - Value: Single scalar in [-1, +1] range

### 🌲 Monte Carlo Tree Search
- **UCB1 selection**: Balances exploration vs exploitation with C_PUCT = 1.5
- **Neural network priors**: Policy predictions guide initial move probabilities
- **Tree expansion**: Dynamically builds search tree based on promising paths
- **Value backpropagation**: Alternating perspective up the tree (-value at each level)

### 🎯 Self-Play Training Loop
- **Data collection**: 15 games per iteration, 30 MCTS simulations per move
- **Training targets**:
  - Policy: Visit count distribution from MCTS
  - Value: Final game outcome from current player's perspective  
- **Experience buffer**: 20,000 most recent positions (deque with maxlen)
- **Model persistence**: Automatic saving after each iteration

## 📊 Current Results

After multiple self-play and training iterations, the model has reached:

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Loss** | 0.4009 | Combined policy + value loss |
| **Policy Accuracy** | ~98% (validation: 100%) | High accuracy, potential overfitting |
| **Value Head MSE** | ~1.29e-09 | Extremely low, may indicate trivial learning |

⚠️ **Important Note**: These metrics are promising but don't yet translate to expert-level play. At this stage, the model mostly imitates its own search process, not the strategic depth of grandmasters.

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install tensorflow numpy python-chess
```

### Quick Start
1. **Run the training pipeline**:
   ```bash
   python MagnusAI.py
   ```

The script will automatically:
- ✅ Try to load existing model (`magnus_model.keras`)
- ✅ Build new dual-head network if no model found
- ✅ Run 10 iterations of self-play (15 games each) + training
- ✅ Save model and move mappings after each iteration

### Training Process
Each iteration consists of:
1. **Self-play**: AI plays 15 complete games against itself
2. **Data collection**: ~300-600 training positions per iteration
3. **Neural network training**: 5 epochs on accumulated data
4. **Model saving**: Preserves progress for next iteration

### Evaluation
Built-in evaluation function tests AI against random moves:
```python
# Example: evaluate current model
ai.evaluate_ai(num_games=10, simulations=100)
# Output: X wins, Y losses, Z draws
```

## 🏗️ Technical Architecture

### Board Representation
- **Input format**: 8×8×12 tensor (6 piece types × 2 colors)
- **FEN parsing**: Converts chess positions to neural network input
- **Move mapping**: 4,096+ UCI strings mapped to output indices

### MCTS Implementation
```python
class MCTSNode:
    - board: chess.Board        # Position state
    - children: dict           # Legal moves → child nodes  
    - visits: int              # Number of simulations through this node
    - value_sum: float         # Accumulated values from simulations
    - prior: float             # Neural network move probability
```

### Training Data Format
Each training example contains:
- **Board state**: 8×8×12 position tensor
- **Policy target**: MCTS visit count distribution
- **Value target**: Game result from current player's perspective

## 🎯 Next Steps

- **Phase 2**: Train on grandmaster games (Magnus Carlsen and others)
- **Phase 3**: Scale reinforcement learning with larger self-play datasets  
- **Final Phase**: Release interactive chess GUI to play against MagnusAI

## 📁 Project Structure

```
MagnusAI-Phase1/
├── MagnusAI.py              # Main training script
├── magnus_model.keras       # Saved neural network (generated)
├── magnus_model_mapping.pkl # Move mappings (generated)
└── README.md               # This file
```

## 🔧 Configuration

Key hyperparameters in `MagnusAI.py`:
```python
# MCTS
C_PUCT = 1.5                    # Exploration constant
simulations_per_move = 30       # MCTS rollouts per move

# Training  
batch_size = 64                 # Training batch size
epochs = 5                      # Training epochs per iteration
learning_rate = 0.001           # Adam optimizer learning rate

# Self-play
num_games = 15                  # Games per iteration
training_data.maxlen = 20000    # Experience buffer size
```

---

