# -*- coding: utf-8 -*-
"""
SMOKE TEST MINI - Agent 7 V2.1 (ULTRA FAST)

Quick test (~1 minute, 100 steps) to verify:
- Model loads
- Takes 3 actions (SELL, HOLD, BUY)
- Opens/closes positions
- Not 100% HOLD or SELL

Duration: ~1 minute
"""

import sys
from pathlib import Path

project_root = Path("C:/Users/lbye3/Desktop/GoldRL")
parent_dir = Path(__file__).resolve().parent.parent  # FICHIER IMPORTANT AGENT 7
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'AGENT_V2'))
sys.path.append(str(parent_dir))  # For organized subdirectories

import numpy as np
from collections import Counter
import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv
from stable_baselines3 import PPO

print("="*60)
print("SMOKE TEST MINI - Agent 7 (100 steps, ~1 min)")
print("="*60)

# Load data
print("[1/3] Loading data...")
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TRAIN_START_DATE, config.TRAIN_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]
print(f"   [OK] {features.shape}")

# Create env
print("[2/3] Creating environment...")
env = GoldTradingEnv(features, prices, config.INITIAL_BALANCE, 'discrete', verbose=False)
print(f"   [OK] Obs: {env.observation_space.shape}")

# Load model
print("[3/3] Loading model...")
models_dir = Path(__file__).parent.parent / 'models'
paths = [
    models_dir / 'agent7_critic_boost_lstm_final.zip',
    models_dir / 'checkpoints' / 'agent7_critic_boost_lstm_500000_steps.zip',
    models_dir / 'checkpoints' / 'agent7_ultimate_500000_steps.zip',
    models_dir / 'checkpoints' / 'agent7_ultimate_400000_steps.zip',
    models_dir / 'best_model.zip'
]

model = None
for p in paths:
    if p.exists():
        try:
            model = PPO.load(p)
            print(f"   [OK] {p.name}")
            break
        except:
            continue

if model is None:
    print("[ERROR] No model found!")
    sys.exit(1)

# Run test
print("\n[TEST] Running 100 steps...")
actions = []
positions_opened = 0
positions_closed = 0

obs = env.reset()
for i in range(100):
    was_flat = (env.position_side == 0)
    action, _ = model.predict(obs, deterministic=False)
    actions.append(action)
    obs, _, done, _, _ = env.step(action)

    if was_flat and env.position_side != 0:
        positions_opened += 1
    if done:
        obs = env.reset()

positions_closed = len(env.trades)

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

counts = Counter(actions)
print(f"\nACTIONS (100 steps):")
print(f"  SELL: {counts.get(0,0):2d} ({counts.get(0,0)}%)")
print(f"  HOLD: {counts.get(1,0):2d} ({counts.get(1,0)}%)")
print(f"  BUY:  {counts.get(2,0):2d} ({counts.get(2,0)}%)")

max_count = max(counts.values())
if max_count > 90:
    print(f"\n[ERROR] MODE COLLAPSE! {max_count}% single action")
elif max_count > 70:
    print(f"\n[WARNING] Strong preference: {max_count}%")
else:
    print(f"\n[OK] Balanced ({max_count}% max)")

print(f"\nPOSITIONS:")
print(f"  Opened: {positions_opened}")
print(f"  Closed: {positions_closed}")

if positions_opened > 0 and positions_closed > 0:
    print(f"\n[OK] Opens AND closes positions")
else:
    print(f"\n[ERROR] Problem with positions!")

print("\n" + "="*60)
if max_count <= 70 and positions_opened > 0 and positions_closed > 0:
    print("[PASS] SMOKE TEST PASSED")
else:
    print("[FAIL] SMOKE TEST FAILED")
print("="*60)
