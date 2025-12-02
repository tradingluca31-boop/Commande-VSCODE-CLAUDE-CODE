# -*- coding: utf-8 -*-
"""
QUICK TEST 10K - Agent 7 V2.1 CRITIC BOOST + LSTM

Fast test (~5 min, 10K steps) to verify:
1. Agent opens AND closes positions
2. Actions are balanced (not 100% single action)
3. Trades are actually executed
4. No mode collapse

This test trains a FRESH model for 10K steps and monitors trading activity.
"""

import sys
from pathlib import Path

# Add project root and parent directory
project_root = Path("C:/Users/lbye3/Desktop/GoldRL")
parent_dir = Path(__file__).resolve().parent.parent  # FICHIER IMPORTANT AGENT 7
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'AGENT_V2'))
sys.path.append(str(parent_dir))

import numpy as np
from collections import Counter
from datetime import datetime
import config

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv

# Use RecurrentPPO for LSTM
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

print("="*70)
print("QUICK TEST 10K - AGENT 7 V2.1 CRITIC BOOST + LSTM")
print("="*70)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[TARGET] Verify agent opens/closes positions")
print("="*70)

# ============================================================================
# CUSTOM CALLBACK TO TRACK TRADING
# ============================================================================
class TradingMonitorCallback(BaseCallback):
    """Monitor trading activity during training"""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.actions = []
        self.trades_opened = 0
        self.trades_closed = 0
        self.last_position = 0

    def _on_step(self) -> bool:
        # Get action from locals (key is 'actions' not 'action')
        if 'actions' in self.locals:
            action = self.locals['actions'][0]
            self.actions.append(int(action))

        # Check for position changes via environment
        try:
            # Access the unwrapped environment
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env

            if hasattr(env, 'position_side'):
                current_position = env.position_side
                if self.last_position == 0 and current_position != 0:
                    self.trades_opened += 1
                elif self.last_position != 0 and current_position == 0:
                    self.trades_closed += 1
                self.last_position = current_position
        except:
            pass

        # Print progress every 2000 steps
        if self.num_timesteps % 2000 == 0:
            action_counts = Counter(self.actions[-2000:]) if len(self.actions) >= 2000 else Counter(self.actions)
            total = sum(action_counts.values())
            if total > 0:
                pcts = {k: v/total*100 for k, v in action_counts.items()}
                print(f"  Step {self.num_timesteps}: SELL {pcts.get(0,0):.1f}% | HOLD {pcts.get(1,0):.1f}% | BUY {pcts.get(2,0):.1f}% | Trades: {self.trades_opened}/{self.trades_closed}")

        return True

    def get_stats(self):
        action_counts = Counter(self.actions)
        total = sum(action_counts.values())
        if total > 0:
            pcts = {k: v/total*100 for k, v in action_counts.items()}
        else:
            pcts = {0: 0, 1: 0, 2: 0}
        return {
            'sell_pct': pcts.get(0, 0),
            'hold_pct': pcts.get(1, 0),
            'buy_pct': pcts.get(2, 0),
            'trades_opened': self.trades_opened,
            'trades_closed': self.trades_closed,
            'total_actions': total
        }

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TRAIN_START_DATE, config.TRAIN_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]
print(f"   [OK] Features: {features.shape}, Prices: {prices.shape}")

# ============================================================================
# 2. CREATE ENVIRONMENT
# ============================================================================
print("\n[2/4] Creating environment...")
env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
print(f"   [OK] Environment created - Obs: {env.observation_space.shape}")

# ============================================================================
# 3. CREATE FRESH MODEL
# ============================================================================
print("\n[3/4] Creating FRESH RecurrentPPO model...")

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=3e-4,
    n_steps=256,  # Smaller for faster feedback
    batch_size=64,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.35,  # V3 AGGRESSIVE: MAXIMUM entropy (0.35) for forcing exploration
    vf_coef=1.0,    # CRITIC BOOST
    max_grad_norm=0.5,
    policy_kwargs=dict(
        lstm_hidden_size=128,  # Smaller for speed
        n_lstm_layers=1,
        enable_critic_lstm=True,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    ),
    verbose=0
)
print("   [OK] Model created with MAXIMUM entropy (0.35) for aggressive exploration")

# ============================================================================
# 4. TRAIN 10K STEPS
# ============================================================================
print("\n[4/4] Training 10K steps (~5 min)...")
print("   Monitoring trading activity...\n")

callback = TradingMonitorCallback()

try:
    model.learn(
        total_timesteps=10_000,
        callback=callback,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("\n   [STOPPED] Training interrupted by user")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*70)
print("QUICK TEST 10K - RESULTS")
print("="*70)

stats = callback.get_stats()

print(f"\n[ACTIONS DISTRIBUTION]")
print(f"   SELL: {stats['sell_pct']:.1f}%")
print(f"   HOLD: {stats['hold_pct']:.1f}%")
print(f"   BUY:  {stats['buy_pct']:.1f}%")

print(f"\n[TRADES]")
print(f"   Opened: {stats['trades_opened']}")
print(f"   Closed: {stats['trades_closed']}")

print(f"\n[VERDICT]")
max_action = max(stats['sell_pct'], stats['hold_pct'], stats['buy_pct'])

if stats['trades_opened'] == 0:
    print("   [ERROR] NO TRADES OPENED - Agent is too passive!")
    print("   -> Check confidence_threshold in environment")
    print("   -> Check if action leads to trade execution")
elif max_action > 90:
    print(f"   [ERROR] MODE COLLAPSE - {max_action:.1f}% single action!")
    print("   -> Increase entropy coefficient")
    print("   -> Check reward function balance")
elif max_action > 70:
    print(f"   [WARNING] Strong preference - {max_action:.1f}% max action")
    print("   -> May need more entropy or reward tuning")
else:
    print(f"   [OK] BALANCED actions ({max_action:.1f}% max)")
    print(f"   [OK] Trades executed: {stats['trades_opened']} opened, {stats['trades_closed']} closed")
    print("   -> Ready for full 500K training!")

print("\n" + "="*70)
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Cleanup
env.close()
