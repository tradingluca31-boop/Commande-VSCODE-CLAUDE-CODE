# -*- coding: utf-8 -*-
"""
================================================================================
FIND BLOCKER - Pourquoi les trades ne s'ouvrent pas?
================================================================================

L'agent choisit BUY/SELL mais aucun trade ne s'ouvre!
Ce script trouve exactement CE QUI BLOQUE.

================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
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

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

print("="*70)
print("🔍 FIND BLOCKER - Pourquoi les trades ne s'ouvrent pas?")
print("="*70)

# ============================================================================
# LOAD
# ============================================================================
print("\n[SETUP] Chargement...")

loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]

# Create environment WITH VERBOSE to see what's happening
env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=True  # IMPORTANT: See all logs!
)

def get_unwrapped_env(vec_env):
    e = vec_env.envs[0]
    while hasattr(e, 'env'):
        e = e.env
    return e

env_monitor = Monitor(env)
vec_env = DummyVecEnv([lambda: env_monitor])

model_paths = [
    parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip",
    parent_dir / "models" / "best_model.zip",
]
model = None
for path in model_paths:
    if path.exists():
        model = RecurrentPPO.load(path, env=vec_env)
        print(f"   Model: {path.name}")
        break

print("   OK\n")

# ============================================================================
# TEST 1: Force BUY action and check what happens
# ============================================================================
print("="*70)
print("TEST 1: FORCER UNE ACTION BUY")
print("="*70)

obs = vec_env.reset()
raw_env = get_unwrapped_env(vec_env)

print(f"\nÉtat initial:")
print(f"   - Position: {raw_env.position_side}")
print(f"   - Balance: ${raw_env.balance:,.2f}")
print(f"   - Drawdown: {raw_env.max_drawdown*100:.2f}%")
print(f"   - Daily loss reached: {raw_env.daily_loss_limit_reached}")
print(f"   - Risk multiplier: {raw_env.risk_multiplier}")

# Check confidence threshold
threshold = raw_env._calculate_min_confidence_threshold()
print(f"   - Confidence threshold: {threshold*100:.1f}%")
print(f"   - Pour discrete, size=100% (toujours > threshold)")

print(f"\n🎯 Je force l'action BUY (action=2)...")
print("-"*50)

# Force BUY
obs, reward, done, info = vec_env.step(np.array([2]))

print("-"*50)
print(f"\nAprès BUY forcé:")
print(f"   - Position: {raw_env.position_side}")
print(f"   - Balance: ${raw_env.balance:,.2f}")
print(f"   - Trades: {len(raw_env.trades)}")

if raw_env.position_side == 0:
    print(f"\n   🚨 TRADE NON OUVERT! Quelque chose bloque!")
else:
    print(f"\n   ✅ Trade ouvert!")

# ============================================================================
# TEST 2: Check _execute_action directly
# ============================================================================
print("\n" + "="*70)
print("TEST 2: ANALYSE DE _execute_action")
print("="*70)

obs = vec_env.reset()
raw_env = get_unwrapped_env(vec_env)

print(f"\nVérification des conditions de blocage:")

# 1. Can open new trade?
can_open = raw_env._can_open_new_trade()
print(f"   1. _can_open_new_trade(): {can_open}")

# 2. Daily loss?
print(f"   2. daily_loss_limit_reached: {raw_env.daily_loss_limit_reached}")

# 3. Max drawdown?
print(f"   3. max_drawdown: {raw_env.max_drawdown*100:.2f}% (limit: 10%)")

# 4. Tail risk?
print(f"   4. tail_risk_detected: {raw_env.tail_risk_detected}")

# 5. Risk multiplier?
print(f"   5. risk_multiplier: {raw_env.risk_multiplier}")

# 6. Confidence threshold?
threshold = raw_env._calculate_min_confidence_threshold()
print(f"   6. min_confidence_threshold: {threshold*100:.1f}%")
print(f"      Pour discrete action, size = 1.0 (100%)")
print(f"      1.0 > {threshold:.2f}? {1.0 > threshold}")

# ============================================================================
# TEST 3: Step-by-step execution analysis
# ============================================================================
print("\n" + "="*70)
print("TEST 3: ANALYSE STEP-BY-STEP")
print("="*70)

obs = vec_env.reset()
raw_env = get_unwrapped_env(vec_env)

print(f"\nJe vais forcer 10 actions BUY et voir ce qui se passe:\n")

for i in range(10):
    pos_before = raw_env.position_side

    # Force BUY
    obs, reward, done, info = vec_env.step(np.array([2]))

    pos_after = raw_env.position_side

    status = "✅ OUVERT!" if pos_after != 0 and pos_before == 0 else "❌ Pas ouvert"
    print(f"   Step {i+1}: BUY forcé | Position: {pos_before} → {pos_after} | {status}")

    if done[0]:
        obs = vec_env.reset()

# ============================================================================
# TEST 4: Check the actual _execute_action code flow
# ============================================================================
print("\n" + "="*70)
print("TEST 4: DÉBOGAGE DE _execute_action")
print("="*70)

obs = vec_env.reset()
raw_env = get_unwrapped_env(vec_env)

print(f"\nSimulation manuelle de _execute_action pour BUY:")
print(f"")

# Simulate what happens in _execute_action
direction = 1  # BUY
size = 1.0  # For discrete

print(f"   direction = {direction} (BUY)")
print(f"   size = {size}")

# Check 1: HOLD check
if abs(direction) < 0.3 or size < 0.1:
    print(f"   ❌ BLOQUÉ: abs(direction) < 0.3 or size < 0.1")
else:
    print(f"   ✅ Pas HOLD (direction={direction}, size={size})")

# Check 2: Close existing position
if raw_env.position_side != 0 and np.sign(direction) != raw_env.position_side:
    print(f"   → Fermeture position existante")
else:
    print(f"   → Pas de position à fermer (position_side={raw_env.position_side})")

# Check 3: Can open new trade
if raw_env.position_side == 0:
    print(f"   → Position est FLAT, on peut ouvrir")

    # Check 3a: _can_open_new_trade
    if not raw_env._can_open_new_trade():
        print(f"   ❌ BLOQUÉ: _can_open_new_trade() = False")
    else:
        print(f"   ✅ _can_open_new_trade() = True")

    # Check 3b: Confidence threshold
    min_conf = raw_env._calculate_min_confidence_threshold()
    if size < min_conf:
        print(f"   ❌ BLOQUÉ: size ({size}) < min_confidence ({min_conf})")
    else:
        print(f"   ✅ size ({size}) >= min_confidence ({min_conf})")

    # Check 3c: Risk multiplier
    raw_env.risk_multiplier = raw_env._calculate_risk_multiplier()
    if raw_env.risk_multiplier <= 0.0:
        print(f"   ❌ BLOQUÉ: risk_multiplier = {raw_env.risk_multiplier}")
    else:
        print(f"   ✅ risk_multiplier = {raw_env.risk_multiplier}")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*70)
print("💡 CONCLUSION")
print("="*70)

print(f"""
   L'agent CHOISIT BUY/SELL mais les trades NE S'OUVRENT PAS.

   Vérifications:
   - _can_open_new_trade(): {can_open}
   - daily_loss_limit_reached: {raw_env.daily_loss_limit_reached}
   - max_drawdown: {raw_env.max_drawdown*100:.2f}%
   - tail_risk_detected: {raw_env.tail_risk_detected}
   - risk_multiplier: {raw_env.risk_multiplier}
   - min_confidence: {threshold*100:.1f}%

   Si tous ces checks sont OK mais pas de trade:
   → Le problème est ailleurs dans le code!

   Prochaine étape: Ajouter des logs dans _execute_action
""")

print("="*70)

vec_env.close()
