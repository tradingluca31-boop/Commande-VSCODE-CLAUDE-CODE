# -*- coding: utf-8 -*-
"""
================================================================================
CHECK POLICY VALUES - Agent 7 V2.1
================================================================================

Regarde ce que l'agent "PENSE" vraiment:
- Pour PPO: Probabilités d'action (policy output)
- Value Function V(s) du Critic
- Comparaison HOLD vs BUY vs SELL

Si P(HOLD) >> P(BUY) et P(SELL) → L'agent a appris que ne rien faire est "safe"
Si V(s) est plat → Le Critic ne distingue pas les bons/mauvais états

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

import torch

print("="*70)
print("CHECK POLICY VALUES - Agent 7 V2.1")
print("Que pense VRAIMENT l'agent?")
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
env = Monitor(env)
vec_env = DummyVecEnv([lambda: env])
print(f"   OK - Obs: {vec_env.observation_space.shape}")

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

if model is None:
    print("   ERREUR: Aucun modèle trouvé!")
    sys.exit(1)

# ============================================================================
# ANALYSE DES POLICY VALUES
# ============================================================================

print("\n" + "="*70)
print("ANALYSE DES POLICY VALUES (ce que l'agent PENSE)")
print("="*70)

# Reset
obs = vec_env.reset()
lstm_states = None

# Storage
all_probs = []
all_values = []
all_actions = []

N_STEPS = 200

print(f"\nAnalyse sur {N_STEPS} steps...\n")
print("-"*70)
print(f"{'Step':>5} | {'P(SELL)':>8} | {'P(HOLD)':>8} | {'P(BUY)':>8} | {'V(s)':>10} | Action")
print("-"*70)

for step in range(N_STEPS):
    # =========================================
    # 1. GET ACTION PROBABILITIES (Policy Output)
    # =========================================
    # Sample multiple times to estimate probabilities
    action_samples = []
    for _ in range(50):
        action, _ = model.predict(obs, state=lstm_states, deterministic=False)
        action_samples.append(int(action[0]))

    # Calculate probabilities
    counts = Counter(action_samples)
    probs = {
        0: counts.get(0, 0) / 50,  # SELL
        1: counts.get(1, 0) / 50,  # HOLD
        2: counts.get(2, 0) / 50   # BUY
    }
    all_probs.append(probs)

    # =========================================
    # 2. GET VALUE FUNCTION V(s) (Critic Output)
    # =========================================
    # For RecurrentPPO, we need to access the value function
    try:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float()
            if hasattr(model.policy, 'device'):
                obs_tensor = obs_tensor.to(model.policy.device)

            # Get value from policy
            # RecurrentPPO stores LSTM states internally
            if lstm_states is not None:
                value = model.policy.predict_values(
                    obs_tensor,
                    lstm_states[0],  # hidden state
                    lstm_states[1],  # cell state
                    torch.zeros(1)   # episode_starts
                )
            else:
                # Initialize LSTM states
                value = torch.tensor([0.0])

            v_s = float(value.cpu().numpy().flatten()[0]) if hasattr(value, 'cpu') else 0.0
    except Exception as e:
        v_s = 0.0

    all_values.append(v_s)

    # =========================================
    # 3. GET DETERMINISTIC ACTION
    # =========================================
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    action_int = int(action[0])
    all_actions.append(action_int)
    action_names = {0: "SELL", 1: "HOLD", 2: "BUY "}

    # Print every 10 steps
    if step % 10 == 0 or step < 20:
        # Highlight if HOLD dominates
        marker = ""
        if probs[1] > 0.8:
            marker = " ⚠️ HOLD DOMINANT!"
        elif probs[1] > 0.6:
            marker = " ⚡"

        print(f"{step:5d} | {probs[0]*100:7.1f}% | {probs[1]*100:7.1f}% | {probs[2]*100:7.1f}% | {v_s:10.4f} | {action_names[action_int]}{marker}")

    # Execute step
    obs, reward, done, info = vec_env.step(action)

    if done[0]:
        obs = vec_env.reset()
        lstm_states = None

print("-"*70)

# ============================================================================
# ANALYSE STATISTIQUE
# ============================================================================

print("\n" + "="*70)
print("STATISTIQUES GLOBALES")
print("="*70)

# Average probabilities
avg_sell = np.mean([p[0] for p in all_probs])
avg_hold = np.mean([p[1] for p in all_probs])
avg_buy = np.mean([p[2] for p in all_probs])

print(f"\n📊 PROBABILITÉS MOYENNES (Policy Output):")
print(f"")
print(f"   SELL: {'█' * int(avg_sell*30):<30} {avg_sell*100:.1f}%")
print(f"   HOLD: {'█' * int(avg_hold*30):<30} {avg_hold*100:.1f}%")
print(f"   BUY:  {'█' * int(avg_buy*30):<30} {avg_buy*100:.1f}%")

# Value function stats
v_mean = np.mean(all_values)
v_std = np.std(all_values)
v_min = np.min(all_values)
v_max = np.max(all_values)

print(f"\n📈 VALUE FUNCTION V(s) (Critic Output):")
print(f"")
print(f"   Moyenne: {v_mean:.4f}")
print(f"   Écart-type: {v_std:.4f}")
print(f"   Min: {v_min:.4f}")
print(f"   Max: {v_max:.4f}")

# Action distribution
action_counts = Counter(all_actions)
print(f"\n🎯 ACTIONS EXÉCUTÉES:")
print(f"")
print(f"   SELL: {action_counts.get(0,0):4d} ({action_counts.get(0,0)/N_STEPS*100:.1f}%)")
print(f"   HOLD: {action_counts.get(1,0):4d} ({action_counts.get(1,0)/N_STEPS*100:.1f}%)")
print(f"   BUY:  {action_counts.get(2,0):4d} ({action_counts.get(2,0)/N_STEPS*100:.1f}%)")

# ============================================================================
# DIAGNOSTIC
# ============================================================================

print("\n" + "="*70)
print("🔍 DIAGNOSTIC")
print("="*70)

problems = []

# Check if HOLD dominates
if avg_hold > 0.7:
    problems.append(f"❌ P(HOLD) = {avg_hold*100:.0f}% >> P(BUY) et P(SELL)")
    problems.append(f"   → L'agent a appris que HOLD est 'SAFE'")
    problems.append(f"   → Il évite le risque en ne tradant pas")

if avg_hold > 0.9:
    problems.append(f"🚨 MODE COLLAPSE SÉVÈRE: {avg_hold*100:.0f}% HOLD!")

# Check value function
if v_std < 0.01:
    problems.append(f"❌ V(s) std = {v_std:.4f} (TROP PLAT)")
    problems.append(f"   → Le Critic ne distingue pas les états")
    problems.append(f"   → L'agent ne voit pas de différence entre situations")

# Check action diversity
hold_executed = action_counts.get(1, 0) / N_STEPS
if hold_executed > 0.9:
    problems.append(f"❌ {hold_executed*100:.0f}% des actions sont HOLD")
    problems.append(f"   → L'agent est PARALYSÉ")

if problems:
    print(f"\n🚨 PROBLÈMES DÉTECTÉS:\n")
    for p in problems:
        print(f"   {p}")
else:
    print(f"\n✅ Pas de problème majeur détecté")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "="*70)
print("💡 CONCLUSION")
print("="*70)

if avg_hold > 0.7:
    print(f"""
   L'AGENT A PEUR DE TRADER!

   Ce qu'il "pense":
   - "HOLD est safe, je ne perds pas d'argent"
   - "BUY et SELL sont risqués, je pourrais perdre"
   - "Mieux vaut ne rien faire que de risquer"

   Pourquoi:
   - L'entropy a diminué trop vite pendant le training
   - Les premiers trades étaient peut-être perdants
   - Le reward pour HOLD ≈ 0 (pas de pénalité)
   - Le reward pour BUY/SELL = risque de perte

   Solutions appliquées (V2):
   ✅ Entropy étendue: 0-50% à 0.25
   ✅ Entropy minimum: 0.12 (pas 0.05)
   ✅ Bonus +0.03 pour BUY/SELL
   ✅ Pénalité -0.02 pour HOLD excessif

   PROCHAINE ÉTAPE:
   Relancer le training avec les nouveaux paramètres!
""")
else:
    print(f"""
   L'agent semble avoir une distribution équilibrée.

   Vérifier les autres causes:
   - Blocages environnement (daily loss, drawdown)
   - Seuil de confiance trop élevé
   - Problème dans l'exécution des trades
""")

print("="*70)
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

vec_env.close()
