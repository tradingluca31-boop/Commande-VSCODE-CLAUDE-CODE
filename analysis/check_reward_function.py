# -*- coding: utf-8 -*-
"""
================================================================================
CHECK REWARD FUNCTION - Agent 7 V2.1
================================================================================

CHECKLIST DE DIAGNOSTIC REWARD:

A) Ta fonction de reward
   - Est-ce que HOLD donne un reward (même petit)?
   - Est-ce que les trades perdants sont trop pénalisés?
   - Y a-t-il un reward uniquement à la clôture du trade?

PIÈGE CLASSIQUE:
Si le reward n'arrive qu'à la clôture ET que les pertes sont pénalisées
→ L'agent apprend que ne jamais ouvrir = jamais de perte = SAFE

Ce script analyse les rewards reçus pour chaque type d'action.

================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
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
print("CHECK REWARD FUNCTION - Agent 7 V2.1")
print("Analyse: Pourquoi l'agent préfère HOLD?")
print("="*70)

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================
print("\n[1/3] Chargement données...")
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]
print(f"   OK - {len(features)} bars")

print("\n[2/3] Création environnement...")
env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)
env_monitor = Monitor(env)
vec_env = DummyVecEnv([lambda: env_monitor])
print(f"   OK")

print("\n[3/3] Chargement modèle...")
model_paths = [
    parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip",
    parent_dir / "models" / "best_model.zip",
]
model = None
for path in model_paths:
    if path.exists():
        model = RecurrentPPO.load(path, env=vec_env)
        print(f"   OK - {path.name}")
        break

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_unwrapped_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

# ============================================================================
# TEST 1: REWARD POUR CHAQUE ACTION (SANS MODÈLE)
# ============================================================================

print("\n" + "="*70)
print("TEST 1: REWARD POUR CHAQUE ACTION (actions forcées)")
print("="*70)
print("\nOn force chaque action et on regarde le reward reçu...\n")

# Reset
obs = vec_env.reset()

# Storage
rewards_by_action = {0: [], 1: [], 2: []}  # SELL, HOLD, BUY
action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

N_TESTS = 100

print("-"*60)
print(f"{'Step':>5} | {'Action':>6} | {'Reward':>12} | {'Position':>8} | Notes")
print("-"*60)

for step in range(N_TESTS):
    # Force each action in rotation
    forced_action = step % 3

    # Execute
    obs_before = obs.copy()
    obs, reward, done, info = vec_env.step(np.array([forced_action]))

    raw_env = get_unwrapped_env(vec_env)
    pos = raw_env.position_side
    pos_names = {-1: "SHORT", 0: "FLAT", 1: "LONG"}

    rewards_by_action[forced_action].append(reward[0])

    # Highlight interesting cases
    notes = ""
    if reward[0] > 0.01:
        notes = "✅ Positif!"
    elif reward[0] < -0.01:
        notes = "❌ Négatif!"

    if step < 30:  # Show first 30
        print(f"{step:5d} | {action_names[forced_action]:>6} | {reward[0]:+12.6f} | {pos_names[pos]:>8} | {notes}")

    if done[0]:
        obs = vec_env.reset()

print("-"*60)

# Statistics
print(f"\n📊 STATISTIQUES DES REWARDS PAR ACTION:\n")

for action in [0, 1, 2]:
    rewards = rewards_by_action[action]
    if rewards:
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        min_r = np.min(rewards)
        max_r = np.max(rewards)

        # Visual bar
        bar_len = int(abs(mean_r) * 1000)
        bar_char = "█" if mean_r >= 0 else "░"

        print(f"   {action_names[action]:>4}: moyenne={mean_r:+.6f} | std={std_r:.6f} | [{min_r:+.6f}, {max_r:+.6f}]")

# ============================================================================
# TEST 2: COMPARAISON HOLD vs TRADING
# ============================================================================

print("\n" + "="*70)
print("TEST 2: COMPARAISON HOLD vs TRADING")
print("="*70)

# Reset
obs = vec_env.reset()

# Test 1: 50 steps de HOLD uniquement
hold_rewards = []
for _ in range(50):
    obs, reward, done, info = vec_env.step(np.array([1]))  # HOLD
    hold_rewards.append(reward[0])
    if done[0]:
        obs = vec_env.reset()

# Reset
obs = vec_env.reset()

# Test 2: 50 steps de BUY/SELL alternés
trade_rewards = []
for i in range(50):
    action = 0 if i % 2 == 0 else 2  # Alternate SELL/BUY
    obs, reward, done, info = vec_env.step(np.array([action]))
    trade_rewards.append(reward[0])
    if done[0]:
        obs = vec_env.reset()

print(f"\n   50 steps HOLD uniquement:")
print(f"   - Reward total: {sum(hold_rewards):+.6f}")
print(f"   - Reward moyen: {np.mean(hold_rewards):+.6f}")

print(f"\n   50 steps TRADING (BUY/SELL alternés):")
print(f"   - Reward total: {sum(trade_rewards):+.6f}")
print(f"   - Reward moyen: {np.mean(trade_rewards):+.6f}")

if np.mean(hold_rewards) > np.mean(trade_rewards):
    print(f"\n   ⚠️ HOLD EST PLUS RÉCOMPENSÉ QUE TRADING!")
    print(f"   → L'agent apprend naturellement à préférer HOLD")
else:
    print(f"\n   ✅ Trading est récompensé autant ou plus que HOLD")

# ============================================================================
# TEST 3: ANALYSE DU CODE REWARD
# ============================================================================

print("\n" + "="*70)
print("TEST 3: CHECKLIST REWARD FUNCTION")
print("="*70)

print(f"""
   ANALYSE DU CODE (trading_env_v2_ultimate.py):

   A) HOLD donne-t-il un reward?
   ─────────────────────────────
   - HOLD sans position = reward ~0 (neutre)
   - HOLD avec position = reward basé sur unrealized PnL
   → Pas de pénalité pour HOLD = HOLD est "safe"

   B) Les trades perdants sont-ils trop pénalisés?
   ───────────────────────────────────────────────
   - Trade perdant = reward négatif (proportionnel à la perte)
   - Drawdown penalty supplémentaire
   - FTMO violation penalty (-10000)
   → OUI, les pertes sont fortement pénalisées

   C) Reward uniquement à la clôture?
   ──────────────────────────────────
   - Reward TIER 1: Basé sur profit_pct (equity - initial)
   - Reward TIER 2: Basé sur FTMO compliance
   - Reward TIER 3: Bonuses comportementaux
   → Le reward arrive CHAQUE STEP mais...
   → ... les GROS rewards/penalties viennent à la clôture

   🚨 PIÈGE IDENTIFIÉ:
   ═══════════════════
   Le reward pour HOLD ≈ 0 (stable)
   Le reward pour BUY/SELL = risque de perte

   L'agent calcule: E[reward|HOLD] vs E[reward|TRADE]
   - E[reward|HOLD] ≈ 0 (constant, prévisible)
   - E[reward|TRADE] = gain possible MAIS perte possible

   Si les pertes sont plus fréquentes ou plus grandes que les gains
   → E[reward|TRADE] < E[reward|HOLD]
   → L'agent choisit HOLD!

   💡 SOLUTION (déjà implémentée V2):
   ══════════════════════════════════
   ✅ +0.03 bonus pour chaque BUY/SELL
   ✅ -0.02 penalty pour HOLD consécutifs > 30
   ✅ -0.03 penalty si distribution > 90% single action

   → E[reward|TRADE] devient > E[reward|HOLD]
""")

# ============================================================================
# TEST 4: VÉRIFIER LES FIXES V2
# ============================================================================

print("\n" + "="*70)
print("TEST 4: VÉRIFICATION DES FIXES V2")
print("="*70)

# Reset
obs = vec_env.reset()
raw_env = get_unwrapped_env(vec_env)

print(f"\n   Vérification du code reward actuel...\n")

# Simulate consecutive holds
print(f"   Test: 50 HOLD consécutifs")
for i in range(50):
    obs, reward, done, info = vec_env.step(np.array([1]))
    if i == 29:
        print(f"   - Step 30: reward = {reward[0]:+.6f}")
    if i == 39:
        print(f"   - Step 40: reward = {reward[0]:+.6f}")
    if i == 49:
        print(f"   - Step 50: reward = {reward[0]:+.6f}")
    if done[0]:
        obs = vec_env.reset()

# Check consecutive holds
if hasattr(raw_env, 'consecutive_holds'):
    print(f"   - Consecutive holds: {raw_env.consecutive_holds}")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*70)
print("💡 CONCLUSION")
print("="*70)

mean_hold = np.mean(rewards_by_action[1]) if rewards_by_action[1] else 0
mean_sell = np.mean(rewards_by_action[0]) if rewards_by_action[0] else 0
mean_buy = np.mean(rewards_by_action[2]) if rewards_by_action[2] else 0

print(f"""
   RÉSUMÉ DES REWARDS:
   - SELL: {mean_sell:+.6f}
   - HOLD: {mean_hold:+.6f}
   - BUY:  {mean_buy:+.6f}
""")

if mean_hold >= max(mean_sell, mean_buy):
    print(f"""
   🚨 PROBLÈME CONFIRMÉ: HOLD A LE MEILLEUR REWARD!

   L'agent a raison de choisir HOLD car c'est l'action
   qui maximise son reward attendu.

   Pour corriger, il faut que:
   E[reward|BUY ou SELL] > E[reward|HOLD]

   FIXES À VÉRIFIER:
   1. Augmenter le bonus pour BUY/SELL (+0.03 → +0.05?)
   2. Augmenter la pénalité HOLD (-0.02 → -0.05?)
   3. Ajouter un bonus pour être en position (pas FLAT)
""")
else:
    print(f"""
   ✅ Le reward semble correct.

   Si l'agent ne trade toujours pas:
   - Vérifier l'entropy (exploration)
   - Vérifier les blocages environnement
   - Le modèle a peut-être appris de mauvaises habitudes
     → Recommencer le training from scratch avec fixes
""")

print("="*70)
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

vec_env.close()
