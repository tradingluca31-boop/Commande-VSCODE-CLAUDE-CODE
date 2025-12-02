# -*- coding: utf-8 -*-
"""
SMOKE TEST - Agent 7 V2.1 CRITIC BOOST + LSTM

Quick test (~10 min) to verify:
1. Agent loads correctly
2. Takes 3 actions (SELL, HOLD, BUY)
3. Opens AND closes positions
4. No mode collapse (100% single action)

Duration: ~10 minutes (1000 steps)
"""

import sys
from pathlib import Path

# Add project root and parent directory
project_root = Path("C:/Users/lbye3/Desktop/GoldRL")
parent_dir = Path(__file__).resolve().parent.parent  # FICHIER IMPORTANT AGENT 7
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'AGENT_V2'))
sys.path.append(str(parent_dir))  # For organized subdirectories

import numpy as np
from collections import Counter
from datetime import datetime
import config

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv
from stable_baselines3 import PPO

print("="*80)
print("SMOKE TEST - AGENT 7 V2.1 CRITIC BOOST")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[TARGET] Quick verification (1000 steps, ~10 min)")
print("  [1] Model loads correctly")
print("  [2] Takes all 3 actions (SELL, HOLD, BUY)")
print("  [3] Opens AND closes positions")
print("  [4] No mode collapse (100% single action)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")

loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()

fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features_train = fe.compute_all_features(
    start_date=config.TRAIN_START_DATE,
    end_date=config.TRAIN_END_DATE
)

xauusd_h1 = auxiliary_data['xauusd_raw']['H1']
prices_train = xauusd_h1.loc[features_train.index, ['open', 'high', 'low', 'close']]

print(f"  [OK] Data loaded: {features_train.shape}")
print(f"  [INFO] Observation space: 209 base + 20 RL = 229 features")

# ============================================================================
# 2. CREATE ENVIRONMENT
# ============================================================================
print("\n[2/4] Creating environment...")

env = GoldTradingEnv(
    features_df=features_train,
    prices_df=prices_train,
    initial_balance=config.INITIAL_BALANCE,
    action_space_type='discrete',
    verbose=False
)

print(f"  [OK] Environment created")
print(f"     Observation space: {env.observation_space.shape}")
print(f"     Action space: Discrete(3) - [0=SELL, 1=HOLD, 2=BUY]")

# ============================================================================
# 3. LOAD MODEL
# ============================================================================
print("\n[3/4] Loading model...")

models_dir = Path(__file__).parent.parent / 'models'
checkpoint_paths = [
    models_dir / 'agent7_critic_boost_lstm_final.zip',  # V2.1 final
    models_dir / 'checkpoints' / 'agent7_critic_boost_lstm_500000_steps.zip',  # V2.1 500K
    models_dir / 'checkpoints' / 'agent7_ultimate_500000_steps.zip',  # V2.0 500K
    models_dir / 'checkpoints' / 'agent7_ultimate_400000_steps.zip',  # V2.0 400K
    models_dir / 'best_model.zip'  # Best validation
]

model = None
model_name = None
for checkpoint_path in checkpoint_paths:
    if checkpoint_path.exists():
        try:
            model = PPO.load(checkpoint_path)
            model_name = checkpoint_path.name
            print(f"  [OK] Model loaded: {model_name}")
            break
        except Exception as e:
            print(f"  [WARNING] Failed to load {checkpoint_path.name}: {e}")
            continue

if model is None:
    print("\n[ERROR] No checkpoint found. Tried:")
    for p in checkpoint_paths:
        print(f"  - {p}")
    sys.exit(1)

# ============================================================================
# 4. RUN SMOKE TEST (1000 steps)
# ============================================================================
print("\n[4/4] Running smoke test (1000 steps)...")
print("  Expected duration: ~10 minutes")
print()

# Tracking
actions_taken = []
positions_opened = 0
positions_closed = 0
initial_balance = env.balance

obs = env.reset()
for step in range(1000):
    # Predict action
    action, _ = model.predict(obs, deterministic=False)
    actions_taken.append(action)

    # Track position changes
    was_flat = (env.position_side == 0)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Track if position opened
    if was_flat and env.position_side != 0:
        positions_opened += 1

    # Reset if done
    if done:
        obs = env.reset()

    # Progress every 100 steps
    if (step + 1) % 100 == 0:
        print(f"  [Progress] Step {step + 1}/1000 ({(step+1)/10:.0f}%)")

# Count closed positions
positions_closed = len(env.trades)

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "="*80)
print("SMOKE TEST RESULTS")
print("="*80)

# Action Distribution
action_counter = Counter(actions_taken)
action_pcts = {
    'SELL': action_counter.get(0, 0) / len(actions_taken) * 100,
    'HOLD': action_counter.get(1, 0) / len(actions_taken) * 100,
    'BUY': action_counter.get(2, 0) / len(actions_taken) * 100
}

print("\n[1] ACTION DISTRIBUTION (1000 steps)")
print(f"   SELL: {action_pcts['SELL']:5.1f}%  ({action_counter.get(0, 0)} actions)")
print(f"   HOLD: {action_pcts['HOLD']:5.1f}%  ({action_counter.get(1, 0)} actions)")
print(f"   BUY:  {action_pcts['BUY']:5.1f}%  ({action_counter.get(2, 0)} actions)")

# Check for mode collapse
max_pct = max(action_pcts.values())
if max_pct > 90:
    print(f"\n   [ERROR] MODE COLLAPSE DETECTED! {max(action_pcts, key=action_pcts.get)} = {max_pct:.1f}%")
    mode_collapse = True
elif max_pct > 70:
    print(f"\n   [WARNING] Strong preference detected: {max(action_pcts, key=action_pcts.get)} = {max_pct:.1f}%")
    mode_collapse = False
else:
    print(f"\n   [OK] Balanced distribution (max = {max_pct:.1f}%)")
    mode_collapse = False

# Position Management
print("\n[2] POSITION MANAGEMENT")
print(f"   Positions opened:  {positions_opened}")
print(f"   Positions closed:  {positions_closed}")

if positions_opened == 0:
    print(f"\n   [ERROR] NO POSITIONS OPENED! Agent never enters market.")
    position_ok = False
elif positions_closed == 0:
    print(f"\n   [WARNING] NO POSITIONS CLOSED! Agent opens but doesn't close.")
    position_ok = False
else:
    print(f"\n   [OK] Agent opens AND closes positions")
    position_ok = True

# Trading Performance
final_balance = env.balance
pnl = final_balance - initial_balance
pnl_pct = (pnl / initial_balance) * 100

print("\n[3] TRADING PERFORMANCE (1000 steps)")
print(f"   Initial Balance: ${initial_balance:,.0f}")
print(f"   Final Balance:   ${final_balance:,.0f}")
print(f"   PnL:             ${pnl:+,.0f} ({pnl_pct:+.2f}%)")
print(f"   Total Trades:    {positions_closed}")

if positions_closed > 0:
    wins = [t for t in env.trades if t['pnl'] > 0]
    win_rate = len(wins) / len(env.trades) * 100
    print(f"   Win Rate:        {win_rate:.1f}%")

    if len(wins) > 0 and len(wins) < len(env.trades):
        losses = [t for t in env.trades if t['pnl'] <= 0]
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        print(f"   Profit Factor:   {profit_factor:.2f}")

# ============================================================================
# 6. VERDICT
# ============================================================================
print("\n" + "="*80)
print("VERDICT")
print("="*80)

all_checks_passed = True

# Check 1: Model loads
print("\n[PASS] [1/4] Model loads correctly")

# Check 2: Action diversity
if mode_collapse:
    print("[FAIL] [2/4] Action diversity: FAILED (mode collapse)")
    all_checks_passed = False
elif max_pct > 70:
    print("[WARN] [2/4] Action diversity: WARNING (strong preference)")
else:
    print("[PASS] [2/4] Action diversity: PASSED")

# Check 3: Position management
if not position_ok:
    print("[FAIL] [3/4] Position management: FAILED (doesn't open/close)")
    all_checks_passed = False
else:
    print("[PASS] [3/4] Position management: PASSED")

# Check 4: No crashes
print("[PASS] [4/4] No crashes: PASSED")

# Final verdict
print("\n" + "="*80)
if all_checks_passed:
    print("[PASS] SMOKE TEST PASSED - Agent ready for full training/evaluation")
else:
    print("[FAIL] SMOKE TEST FAILED - Fix issues before continuing")
print("="*80)

print(f"\nModel tested: {model_name}")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[NEXT STEPS]")
if all_checks_passed:
    print("  1. Run full evaluation: python evaluate_agent7.py")
    print("  2. Run interview: RUN_INTERVIEW_AGENT7.bat")
    print("  3. Run SHAP: RUN_SHAP_ANALYSIS.bat")
else:
    print("  1. Check training logs for mode collapse")
    print("  2. Try earlier checkpoint (e.g., 200K, 300K)")
    print("  3. Re-run smoke test with different checkpoint")
