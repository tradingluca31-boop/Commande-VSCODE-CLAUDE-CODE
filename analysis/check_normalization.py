# -*- coding: utf-8 -*-
"""
CHECK NORMALIZATION - Verify all 229 features are properly normalized
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add paths
project_root = Path("C:/Users/lbye3/Desktop/GoldRL")
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'AGENT_V2'))
sys.path.append(str(parent_dir))

import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv

print("="*70)
print("CHECK NORMALIZATION - 229 Features")
print("="*70)

# Load data
print("\n[1/3] Loading data...")
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]
print(f"   Features shape: {features.shape}")

# Create environment
print("\n[2/3] Creating environment...")
env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)

# Get multiple observations
print("\n[3/3] Checking normalization on 100 observations...")
obs = env.reset()[0]
observations = [obs]

for _ in range(99):
    action = env.action_space.sample()
    obs, _, done, _, _ = env.step(action)
    observations.append(obs)
    if done:
        obs = env.reset()[0]

obs_array = np.array(observations)
print(f"   Observations shape: {obs_array.shape}")

# Analyze normalization
print("\n" + "="*70)
print("NORMALIZATION ANALYSIS")
print("="*70)

# Overall stats
print(f"\n[OVERALL STATS]")
print(f"   Min:  {obs_array.min():.4f}")
print(f"   Max:  {obs_array.max():.4f}")
print(f"   Mean: {obs_array.mean():.4f}")
print(f"   Std:  {obs_array.std():.4f}")

# Per-feature stats
print(f"\n[PER-FEATURE STATS]")
feature_means = obs_array.mean(axis=0)
feature_stds = obs_array.std(axis=0)
feature_mins = obs_array.min(axis=0)
feature_maxs = obs_array.max(axis=0)

# Check ranges
in_range = np.sum((feature_mins >= -5.5) & (feature_maxs <= 5.5))
print(f"   Features in range [-5.5, 5.5]: {in_range}/229 ({in_range/229*100:.1f}%)")

# Find problematic features
problematic = []
for i in range(229):
    if feature_mins[i] < -5.5 or feature_maxs[i] > 5.5:
        problematic.append({
            'index': i,
            'min': feature_mins[i],
            'max': feature_maxs[i],
            'mean': feature_means[i],
            'std': feature_stds[i]
        })

if problematic:
    print(f"\n[PROBLEMATIC FEATURES] ({len(problematic)} found)")
    for p in problematic[:10]:  # Show first 10
        print(f"   Feature {p['index']}: min={p['min']:.2f}, max={p['max']:.2f}, mean={p['mean']:.2f}, std={p['std']:.2f}")
else:
    print(f"\n   All 229 features are properly normalized!")

# Base vs RL features
print(f"\n[BASE FEATURES (0-208)]")
base_obs = obs_array[:, :209]
print(f"   Min:  {base_obs.min():.4f}")
print(f"   Max:  {base_obs.max():.4f}")
print(f"   Mean: {base_obs.mean():.4f}")
print(f"   Std:  {base_obs.std():.4f}")

print(f"\n[RL FEATURES (209-228)]")
rl_obs = obs_array[:, 209:]
print(f"   Min:  {rl_obs.min():.4f}")
print(f"   Max:  {rl_obs.max():.4f}")
print(f"   Mean: {rl_obs.mean():.4f}")
print(f"   Std:  {rl_obs.std():.4f}")

# Detailed RL features
rl_names = [
    "last_action_0 (SELL)", "last_action_1 (HOLD)", "last_action_2 (BUY)",
    "regret_signal", "position_duration", "unrealized_pnl",
    "regime_0", "regime_1", "regime_2",
    "hours_until_event", "volatility_pct", "position_side",
    "trade_similarity", "win_rate_20", "streak",
    "avg_pnl_20", "best_trade_20", "worst_trade_20",
    "win_count_20", "loss_count_20"
]

print(f"\n[RL FEATURES DETAIL]")
print(f"   {'Feature':<25} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for i, name in enumerate(rl_names):
    idx = 209 + i
    print(f"   {name:<25} {feature_mins[idx]:>8.3f} {feature_maxs[idx]:>8.3f} {feature_means[idx]:>8.3f} {feature_stds[idx]:>8.3f}")

# Verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

all_ok = len(problematic) == 0
base_ok = base_obs.min() >= -5.5 and base_obs.max() <= 5.5
rl_ok = rl_obs.min() >= -5.5 and rl_obs.max() <= 5.5

if all_ok:
    print("\n   [OK] All 229 features are properly normalized!")
    print("   [OK] Base features (209): Z-score per feature, clipped [-5, 5]")
    print("   [OK] RL features (20): Individual normalization")
else:
    print(f"\n   [WARNING] {len(problematic)} features out of range")
    print("   Check the problematic features above")

print("\n" + "="*70)

env.close()
