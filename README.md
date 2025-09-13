# MagnusAI 🏆♟️

*A Chess AI trained from scratch using Deep Reinforcement Learning and Monte Carlo Tree Search*

> **"Every chess master was once a beginner."** - Irving Chernev

MagnusAI is an ambitious project to create a world-class chess-playing artificial intelligence from the ground up. Unlike engines that rely on handcrafted evaluation functions, MagnusAI learns chess strategy entirely through self-play and neural networks, similar to how AlphaZero revolutionized game AI.

## 🎯 Project Vision

The goal is to build an AI that:
- ✨ **Learns from scratch** - No human chess knowledge programmed in
- 🧠 **Develops intuition** - Neural networks discover patterns and strategy
- 🔄 **Self-improves** - Gets stronger by playing millions of games against itself
- 🎮 **Provides interaction** - Playable interface for human opponents
- 📈 **Achieves mastery** - Ultimately reaches grandmaster-level play

## 🚀 Project Roadmap

### Phase 1: Foundation ✅ *[CURRENT]*
**Self-Play Training System**
- [x] Dual-head neural network (policy + value)
- [x] Monte Carlo Tree Search implementation  
- [x] Self-play training loop
- [x] Basic model persistence and evaluation
- **Status**: Training loss ~0.4, ready for enhancement

### Phase 2: Knowledge Integration 🔄 *[NEXT]*  
**Grandmaster Game Training**
- [ ] Database of Magnus Carlsen games
- [ ] Supervised learning on expert positions
- [ ] Combined self-play + supervised training
- [ ] Advanced position evaluation metrics

### Phase 3: Scaling & Optimization ⏳ *[PLANNED]*
**Large-Scale Reinforcement Learning**
- [ ] Distributed self-play across multiple processes
- [ ] Larger neural network architectures
- [ ] Advanced MCTS optimizations (virtual loss, etc.)
- [ ] Opening book and endgame tablebase integration

### Phase 4: Interactive Release 🎮 *[FINAL]*
**Public Chess Interface**
- [ ] Real-time chess GUI with drag-and-drop
- [ ] Multiple difficulty levels
- [ ] Game analysis and move suggestions
- [ ] ELO rating system and progress tracking

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chess Board   │───▶│  Neural Network │───▶│  Move Selection │
│   (8x8x12)      │    │  Policy + Value │    │   (via MCTS)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │              ┌─────────────────┐            │
         └──────────────│   Self-Play     │◀───────────┘
                        │ Training Loop   │
                        └─────────────────┘
```

### Core Components

🧠 **Neural Network**
- **Input**: 8×8×12 board representation (pieces + colors)
- **Architecture**: Convolutional layers with residual connections
- **Outputs**: Move probability distribution + position evaluation
- **Framework**: TensorFlow/Keras with Adam optimizer

🌲 **Monte Carlo Tree Search**  
- **Selection**: UCB1 formula balances exploration vs exploitation
- **Expansion**: Neural network guides promising move exploration
- **Simulation**: Value network estimates position strength
- **Backpropagation**: Results flow back up the search tree

🎯 **Training Pipeline**
- **Self-play**: AI plays complete games against itself
- **Data generation**: Positions, move probabilities, game outcomes
- **Model updates**: Neural network learns from accumulated experience
- **Iteration**: Continuous improvement through repeated cycles

## 📊 Current Performance

### Phase 1 Metrics
| Metric | Value | Target |
|--------|-------|--------|
| Training Loss | 0.4009 | < 0.3 |
| Policy Accuracy | 98% | > 95% |
| Games vs Random | 85% win rate | > 90% |
| Training Speed | 15 games/iteration | 100+ games |

### Roadmap Targets
- **Phase 2**: Beat intermediate players (1400-1600 ELO)
- **Phase 3**: Reach expert level (1800-2000 ELO)  
- **Phase 4**: Achieve master strength (2200+ ELO)

## 🛠️ Getting Started

### Prerequisites
```bash
# Core dependencies
pip install tensorflow numpy python-chess

# Optional: for distributed training (Phase 3)
pip install ray multiprocessing

# Optional: for GUI (Phase 4)  
pip install pygame tkinter
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/MagnusAI.git
cd MagnusAI

# Run Phase 1 training
cd "Phase 1"
python MagnusAI.py

# Monitor training progress
# Watch for decreasing loss and improving win rates
```

### Project Structure
```
MagnusAI/
├── Phase 1/
│   ├── MagnusAI.py              # Self-play training system
│   ├── magnus_model.keras       # Trained neural network
│   └── README.md               # Phase 1 documentation
├── Phase 2/                    # [Coming Soon]
│   └── grandmaster_training.py
├── Phase 3/                    # [Planned]  
│   └── distributed_selfplay.py
├── Phase 4/                    # [Final]
│   └── chess_gui.py
├── data/
│   ├── selfplay_games/         # Generated training games
│   └── grandmaster_pgns/       # Expert game database
├── models/
│   ├── checkpoints/            # Training snapshots
│   └── best_models/            # Top-performing versions
├── docs/
│   ├── architecture.md         # Technical deep-dive
│   ├── training_logs.md        # Performance tracking
│   └── research_notes.md       # Insights and discoveries
└── README.md                   # This file
```

## 🔬 Technical Deep Dive

### Neural Network Architecture
- **Input Processing**: FEN → 8×8×12 tensor transformation
- **Feature Extraction**: Multiple 3×3 convolutional layers with batch normalization
- **Residual Connections**: Skip connections prevent vanishing gradients
- **Policy Head**: Dense layers → softmax over all legal moves (~4,096 outputs)
- **Value Head**: Dense layers → tanh activation for position evaluation [-1, +1]

### MCTS Algorithm Details
```python
# UCB1 Selection Formula
UCB1 = Q(s,a) + C * P(s,a) * √(N(s)) / (1 + N(s,a))

# Where:
# Q(s,a) = Average value of action a from state s
# P(s,a) = Neural network prior probability  
# N(s) = Visit count of state s
# C = Exploration constant (typically 1.0-2.0)
```

### Training Data Format
Each self-play game generates training examples:
- **Position**: Board state as 8×8×12 tensor
- **Policy Target**: MCTS visit count distribution  
- **Value Target**: Final game result from current player's perspective

## 📈 Research Insights

### Discoveries So Far
- **High policy accuracy** doesn't immediately translate to strong play
- **Self-play convergence** requires careful temperature scheduling
- **Value network** learns faster than policy network in early phases
- **MCTS simulations** vs **network quality** trade-off is crucial

### Open Questions
- How many self-play games needed to reach expert level?
- Optimal neural network architecture for chess?
- Best way to integrate opening theory and endgame knowledge?
- Can we achieve AlphaZero performance with less compute?

## 🤝 Contributing

We welcome contributions from chess players, AI researchers, and developers!

### Ways to Help
- 🐛 **Bug fixes**: Improve training stability and performance
- 🎨 **GUI development**: Create beautiful chess interfaces  
- 📊 **Analysis tools**: Visualize training progress and game quality
- 🏗️ **Architecture**: Experiment with network designs
- 📚 **Documentation**: Improve guides and explanations

### Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/yourusername/MagnusAI.git

# Create feature branch
git checkout -b feature/your-improvement

# Make changes and test
python -m pytest tests/

# Submit pull request
git push origin feature/your-improvement
```

## 📚 Resources & References

### Chess AI Research
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) - AlphaZero paper
- [Deep Blue vs Kasparov](https://en.wikipedia.org/wiki/Deep_Blue_versus_Garry_Kasparov) - Historic human vs AI match
- [Leela Chess Zero](https://lczero.org/) - Open-source neural network chess engine

### Technical Resources  
- [Python Chess Library](https://python-chess.readthedocs.io/) - Chess game logic and utilities
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) - Neural network implementation
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) - Algorithm overview

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **Magnus Carlsen** - Inspiration for the project name and chess excellence
- **DeepMind Team** - Pioneering work on AlphaZero and game AI
- **Chess.com Community** - Providing game databases and testing opportunities
- **Open Source Contributors** - Making tools like python-chess and TensorFlow available

---

**Current Status**: Phase 1 Complete ✅ | **Next Milestone**: Grandmaster Training Integration 🎯
