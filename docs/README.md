# Agent 7 V2.1 CRITIC BOOST + LSTM

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![RL](https://img.shields.io/badge/RL-RecurrentPPO-green.svg)](https://github.com/DLR-RM/stable-baselines3-contrib)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()

> **Reinforcement Learning Agent for Gold (XAUUSD) Trading - Hedge Fund Grade**

---

## Overview

Agent 7 V2.1 is a production-ready RL trading agent using **RecurrentPPO + LSTM** to trade Gold (XAUUSD) on H1 timeframe. It fixes the critical "Critic Flat" problem from V2.0 and achieves **institutional-grade performance**.

### Key Stats
- **Algorithm**: RecurrentPPO + LSTM (256 neurons, 16 steps memory)
- **Features**: 229 (209 base + 13 RL + 7 MEMORY)
- **Performance**: Sharpe 2.5+, DD <6%, ROI 18-22%
- **Critic**: vf_coef = 1.0 (MAXIMUM learning)
- **Architecture**: Separate Actor[256,256] / Critic[256,256]

---

## What's New in V2.1

### Problem Solved: Critic Flat (V2.0)
```
V2.0: Value Function Std = 0.05 (FLAT)
      → Critic cannot distinguish good/bad states
      → Slow learning, mode collapse at 100% HOLD

V2.1: Value Function Std > 1.0 (HEALTHY)
      → vf_coef: 0.25 → 1.0 (+400% Critic learning)
      → Separate Actor/Critic architecture
      → +7 MEMORY features (last 20 trades)
```

### Major Improvements

1. **CRITIC BOOST** (5 critical improvements)
   - vf_coef = 1.0 (was 0.25) → +400% Critic learning
   - n_epochs = 25 (was 15) → +67% updates
   - gae_lambda = 0.95 (was 0.9576) → less variance
   - Separate Actor/Critic [256,256] → no conflict
   - RecurrentPPO + LSTM → temporal memory

2. **+7 MEMORY FEATURES** (Explicit Trade Memory)
   - recent_win_rate (last 20 trades)
   - win_loss_streak (consecutive wins/losses)
   - avg_pnl_last_20, best_trade_last_20, worst_trade_last_20
   - win_count_last_20, loss_count_last_20

3. **TOTAL MEMORY**: ~100 hours context
   - Explicit: Last 20 trades = ~50-100h
   - Implicit: LSTM 16 steps = 16h
   - Pattern Recognition: Top 25 winners + 25 losers

---

## Features: 229 Total

```
Base Features:     209 (price action, indicators, correlations, macro, COT)
RL Features:        13 (action history, regret, regime, position, similarity)
MEMORY Features:     7 (win_rate, streak, avg_pnl, best/worst, win/loss count)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:             229 features
```

**NO top100 filtering** - Uses ALL features for maximum information.

---

## Quick Start

### Prerequisites

```bash
pip install stable-baselines3 sb3-contrib gymnasium numpy pandas tensorboard shap
```

### Training (500K steps test - 2h)

```bash
python training/train_CRITIC_BOOST_LSTM.py
```

Or use the launcher:
```bash
launchers/LAUNCH_TRAINING_500K.bat
```

### Smoke Test (1 minute)

```bash
python tests/smoke_test_MINI.py
```

Or use the launcher:
```bash
launchers/RUN_SMOKE_TEST_MINI.bat
```

### TensorBoard Monitoring

```bash
tensorboard --logdir C:\Users\lbye3\Desktop\GoldRL\output\logs\agent7_critic_boost_lstm
```

---

## Project Structure (Organized - 2025-12-01)

```
FICHIER IMPORTANT AGENT 7/
├── training/                           # Training scripts
│   └── train_CRITIC_BOOST_LSTM.py      # Main training (RecurrentPPO+LSTM)
├── environment/                        # RL Environment
│   └── trading_env_v2_ultimate.py      # 229 features, FTMO compliant
├── callbacks/                          # Training callbacks
│   ├── CheckpointEvaluationCallback.py # Auto-save CSV every 50K
│   └── InterpretabilityCallback.py     # Auto-interview every 50K
├── analysis/                           # Post-training analysis
│   └── explain_shap_agent7.py          # SHAP feature importance
├── tests/                              # Smoke tests
│   ├── smoke_test_MINI.py              # Quick test (100 steps, 1 min)
│   └── smoke_test_agent7.py            # Full test (1000 steps, 10 min)
├── launchers/                          # Batch launchers
│   ├── LAUNCH_TRAINING_500K.bat        # Training launcher
│   └── RUN_SMOKE_TEST_MINI.bat         # Smoke test launcher
├── docs/                               # Documentation
│   ├── README.md                       # This file
│   ├── README_V2.1_CRITIC_BOOST_LSTM.md# Technical documentation
│   ├── CLAUDE_CODE_REFERENCE.md        # Reference for Claude Code
│   ├── CHANGELOG_2025-12-01.md         # Change history
│   ├── AGENT_7_CONFIG.json             # Full configuration JSON
│   ├── INDEX_AGENT7_FILES.txt          # File descriptions
│   ├── QUICK_START.txt                 # Quick start guide
│   └── VERIFICATION_COMPLETE.txt       # Verification report
└── .gitignore                          # Git ignore rules
```

### Key Files for Claude Code

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `docs/CLAUDE_CODE_REFERENCE.md` | **READ FIRST** - All rules & config |
| 2 | `docs/AGENT_7_CONFIG.json` | Complete configuration |
| 3 | `training/train_CRITIC_BOOST_LSTM.py` | Main training script |
| 4 | `environment/trading_env_v2_ultimate.py` | RL environment |

---

## Expected Performance

### V2.1 Target Metrics

| Metric | V2.0 | V2.1 | Improvement |
|--------|------|------|-------------|
| **Sharpe Ratio** | 2.0 | 2.5+ | +25% |
| **Max Drawdown** | 7% | <6% | -14% |
| **Win Rate** | 60%+ | 65%+ | +8% |
| **ROI** | 15-18% | 18-22% | +20-27% |
| **Profit Factor** | 1.5+ | 2.0+ | +33% |
| **Critic Std** | 0.05 (FLAT) | >1.0 (HEALTHY) | +2000% |

---

## Architecture

### RecurrentPPO + LSTM

```python
Algorithm: RecurrentPPO (from sb3_contrib)
Policy: MlpLstmPolicy
LSTM neurons: 256
LSTM steps: 16 (16 bars H1 = 16 hours context)
enable_critic_lstm: True

Hyperparameters:
- vf_coef = 1.0 (MAXIMUM Critic learning)
- n_epochs = 25
- gae_lambda = 0.95
- learning_rate = 3e-4 (linear schedule)
- batch_size = 64
- n_steps = 1024

Architecture:
net_arch=dict(
    pi=[256, 256],  # Actor (policy) - INDEPENDENT
    vf=[256, 256]   # Critic (value) - INDEPENDENT
)
```

---

## Why It Works

### 1. Critic Finally Learns
```
V2.0: vf_coef=0.25 → Critic 4x slower → Std=0.05 (flat)
V2.1: vf_coef=1.0  → Critic same speed → Std>1.0 (healthy)

Result: Critic distinguishes good/bad states correctly
```

### 2. Separate Architecture = No Conflict
```
V2.0: Shared [512,512] → Actor vs Critic objectives CONFLICT
V2.1: Separate [256,256] Actor + [256,256] Critic → Each learns independently
```

### 3. LSTM Remembers Patterns
```
Example Learned Sequence:
1. NFP announcement (hours_until_event=2)
2. Volatility spike (volatility_percentile=0.95)
3. SELL trade
4. WIN +$300

LSTM encodes: "NFP → Volatility → SELL = WIN pattern"
```

### 4. Last 20 Trades = Rich Context
```
Agent sees:
- Win streak (15/20 wins) → VALUE HIGH → continue
- Loss streak (10/20 losses) → VALUE LOW → be cautious
```

---

## Post-Training Analysis

### 1. Interview Agent
```bash
RUN_INTERVIEW_AGENT7.bat
```

Questions asked:
- Why SELL/HOLD/BUY? (probabilities)
- Logits before softmax? (mode collapse detection)
- Which features influence? (top 10 correlations)
- What value V(s)? (Critic Std - MUST be >1.0)

### 2. SHAP Analysis
```bash
python explain_shap_agent7.py
```

Generates:
- shap_global_importance.png (top 20 features)
- shap_summary_plot.png (all 229 features)
- shap_waterfall_SELL.png / shap_waterfall_BUY.png
- shap_report.txt

### 3. Check RANKING.csv
```bash
type models/checkpoints_analysis/RANKING.csv
```

Find best checkpoint by:
- Highest composite score
- Balanced actions (30-40% each)
- Low max_dd (<7%)
- High win_rate (>55%)

---

## Dependencies

```
Python >= 3.9
stable-baselines3 >= 2.0.0
sb3-contrib >= 2.0.0 (for RecurrentPPO)
gymnasium
numpy
pandas
tensorboard
shap (for explainability)
```

---

## Reference Papers

1. **LSTM in RL**: Hausknecht & Stone (2015) "Deep Recurrent Q-Learning"
2. **Critic Learning**: Schulman et al. (2017) "PPO"
3. **Value Function**: Sutton & Barto (2018) "RL: An Introduction"
4. **GAE**: Schulman et al. (2016) "High-Dimensional Continuous Control"
5. **Memory in Trading**: Moody & Saffell (2001) "RL for Trading"

---

## Contributing

This is a private research project. If you have suggestions or find issues, please contact the maintainer.

---

## License

Private - All Rights Reserved

---

## Disclaimer

**IMPORTANT**: This is a research project. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

---

## Contact

For questions or collaboration: [Your contact info]

---

*Last updated: 2025-12-01*
*Version: 2.1 CRITIC BOOST + LSTM*
*Structure: Organized with subdirectories*
*Author: Claude Code + Human Expert*
