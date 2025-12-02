# -*- coding: utf-8 -*-
"""
================================================================================
DIAGNOSTIC COMPLET V2 - Agent 7 V2.1
================================================================================

TOUS LES TESTS DE DIAGNOSTIC EN UN SEUL SCRIPT:

A) Distribution des actions
B) Exploration (entropy pour PPO)
C) Test Random vs Trained
D) Vérification des observations (normalisation)
E) Reward cumulé par type d'action
F) Solutions selon diagnostic

================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
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
print("DIAGNOSTIC COMPLET V2 - Agent 7 V2.1")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================
print("\n[SETUP] Chargement...")

loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]
print(f"   Data: {len(features)} bars")

env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)
env_monitor = Monitor(env)
vec_env = DummyVecEnv([lambda: env_monitor])
print(f"   Environment: OK")

model_paths = [
    parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip",
    parent_dir / "models" / "best_model.zip",
]
model = None
model_name = "None"
for path in model_paths:
    if path.exists():
        model = RecurrentPPO.load(path, env=vec_env)
        model_name = path.name
        print(f"   Model: {path.name}")
        break

if model is None:
    print("   Model: NOT FOUND!")

def get_unwrapped_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

# ============================================================================
# TEST A: DISTRIBUTION DES ACTIONS
# ============================================================================
print("\n" + "="*70)
print("TEST A: DISTRIBUTION DES ACTIONS")
print("="*70)

obs = vec_env.reset()
lstm_states = None
action_counts = {0: 0, 1: 0, 2: 0}  # SELL, HOLD, BUY
N_STEPS = 500

for _ in range(N_STEPS):
    if model:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
        action_int = int(action[0])
    else:
        action_int = np.random.randint(0, 3)

    action_counts[action_int] += 1
    obs, reward, done, info = vec_env.step(np.array([action_int]))

    if done[0]:
        obs = vec_env.reset()
        lstm_states = None

print(f"\n   Distribution sur {N_STEPS} steps:")
print(f"")
total = sum(action_counts.values())
for a, count in action_counts.items():
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    names = {0: "SELL", 1: "HOLD", 2: "BUY "}
    print(f"   {names[a]}: {bar:<50} {pct:5.1f}% ({count})")

# Diagnostic
hold_pct = action_counts[1] / total * 100
if hold_pct > 95:
    print(f"\n   🚨 CRITIQUE: {hold_pct:.0f}% HOLD = MODE COLLAPSE SÉVÈRE!")
elif hold_pct > 80:
    print(f"\n   ⚠️ PROBLÈME: {hold_pct:.0f}% HOLD = Exploration insuffisante!")
elif hold_pct > 60:
    print(f"\n   ⚡ ATTENTION: {hold_pct:.0f}% HOLD = Tendance à l'inaction")
else:
    print(f"\n   ✅ OK: Distribution équilibrée")

# ============================================================================
# TEST B: EXPLORATION (ENTROPY POUR PPO)
# ============================================================================
print("\n" + "="*70)
print("TEST B: EXPLORATION (ENTROPY)")
print("="*70)

if model:
    # Get current entropy coefficient
    ent_coef = model.ent_coef
    print(f"\n   Entropy coefficient actuel: {ent_coef}")

    if ent_coef < 0.05:
        print(f"   🚨 TROP BAS! L'agent n'explore plus")
        print(f"   → Devrait être >= 0.10 pour exploration suffisante")
    elif ent_coef < 0.10:
        print(f"   ⚠️ BAS: Exploration limitée")
    else:
        print(f"   ✅ OK: Exploration suffisante")

    # Check policy entropy by sampling
    print(f"\n   Estimation de l'entropie de la policy:")
    obs = vec_env.reset()
    entropies = []

    for _ in range(100):
        # Sample to estimate probabilities
        action_samples = []
        for _ in range(30):
            action, _ = model.predict(obs, state=lstm_states, deterministic=False)
            action_samples.append(int(action[0]))

        counts = Counter(action_samples)
        probs = [counts.get(i, 0) / 30 for i in range(3)]

        # Calculate entropy: -sum(p * log(p))
        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)

        entropies.append(entropy)

    mean_entropy = np.mean(entropies)
    max_entropy = np.log(3)  # Maximum for 3 actions

    print(f"   - Entropie moyenne: {mean_entropy:.3f}")
    print(f"   - Entropie max possible: {max_entropy:.3f}")
    print(f"   - Ratio: {mean_entropy/max_entropy*100:.1f}%")

    if mean_entropy / max_entropy < 0.3:
        print(f"   🚨 ENTROPIE TRÈS BASSE: L'agent est déterministe!")
    elif mean_entropy / max_entropy < 0.5:
        print(f"   ⚠️ ENTROPIE BASSE: Peu de diversité")
    else:
        print(f"   ✅ OK: Diversité dans les actions")
else:
    print(f"   [SKIP] Pas de modèle chargé")

# ============================================================================
# TEST C: RANDOM VS TRAINED
# ============================================================================
print("\n" + "="*70)
print("TEST C: RANDOM VS TRAINED")
print("="*70)

N_EPISODES = 5
STEPS_PER_EPISODE = 200

# Random agent
print(f"\n   Running Random Agent ({N_EPISODES} episodes x {STEPS_PER_EPISODE} steps)...")
random_rewards = []
for ep in range(N_EPISODES):
    obs = vec_env.reset()
    ep_reward = 0
    for _ in range(STEPS_PER_EPISODE):
        action = np.random.randint(0, 3)
        obs, reward, done, info = vec_env.step(np.array([action]))
        ep_reward += reward[0]
        if done[0]:
            break
    random_rewards.append(ep_reward)

random_mean = np.mean(random_rewards)
random_std = np.std(random_rewards)
print(f"   Random: mean={random_mean:.4f}, std={random_std:.4f}")

# Trained agent
if model:
    print(f"   Running Trained Agent ({N_EPISODES} episodes x {STEPS_PER_EPISODE} steps)...")
    trained_rewards = []
    for ep in range(N_EPISODES):
        obs = vec_env.reset()
        lstm_states = None
        ep_reward = 0
        for _ in range(STEPS_PER_EPISODE):
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]
            if done[0]:
                break
        trained_rewards.append(ep_reward)

    trained_mean = np.mean(trained_rewards)
    trained_std = np.std(trained_rewards)
    print(f"   Trained: mean={trained_mean:.4f}, std={trained_std:.4f}")

    # Compare
    print(f"\n   Comparaison:")
    diff = trained_mean - random_mean
    if abs(diff) < 0.01:
        print(f"   🚨 PROBLÈME: Random ≈ Trained ({diff:+.4f})")
        print(f"   → L'agent n'a RIEN APPRIS d'utile!")
    elif trained_mean > random_mean:
        print(f"   ✅ Trained > Random de {diff:+.4f}")
        print(f"   → L'agent a appris quelque chose")
    else:
        print(f"   ⚠️ Random > Trained de {-diff:+.4f}")
        print(f"   → L'agent fait PIRE que random!")
else:
    print(f"   [SKIP] Pas de modèle chargé")

# ============================================================================
# TEST D: VÉRIFICATION DES OBSERVATIONS (NORMALISATION)
# ============================================================================
print("\n" + "="*70)
print("TEST D: VÉRIFICATION DES OBSERVATIONS")
print("="*70)

obs = vec_env.reset()
obs_flat = obs.flatten()

print(f"\n   Statistiques de l'observation ({len(obs_flat)} features):")
print(f"")
print(f"   Min:  {obs_flat.min():.4f}")
print(f"   Max:  {obs_flat.max():.4f}")
print(f"   Mean: {obs_flat.mean():.4f}")
print(f"   Std:  {obs_flat.std():.4f}")

# Check for problems
problems = []

if abs(obs_flat.max()) > 100:
    problems.append(f"⚠️ Max = {obs_flat.max():.2f} (trop grand!)")

if abs(obs_flat.min()) > 100:
    problems.append(f"⚠️ Min = {obs_flat.min():.2f} (trop petit!)")

if obs_flat.std() > 10:
    problems.append(f"⚠️ Std = {obs_flat.std():.2f} (variance trop haute!)")

if np.isnan(obs_flat).any():
    problems.append(f"🚨 NaN détectés dans les observations!")

if np.isinf(obs_flat).any():
    problems.append(f"🚨 Inf détectés dans les observations!")

if problems:
    print(f"\n   Problèmes détectés:")
    for p in problems:
        print(f"   - {p}")
    print(f"\n   → Les features devraient être normalisées [-3, 3] environ")
else:
    print(f"\n   ✅ Observations bien normalisées")

# Show extreme values
extreme_indices = np.argsort(np.abs(obs_flat))[::-1][:5]
print(f"\n   Top 5 features extrêmes:")
for idx in extreme_indices:
    print(f"   - Feature[{idx}]: {obs_flat[idx]:.4f}")

# ============================================================================
# TEST E: REWARD CUMULÉ PAR TYPE D'ACTION
# ============================================================================
print("\n" + "="*70)
print("TEST E: REWARD PAR TYPE D'ACTION")
print("="*70)

obs = vec_env.reset()
reward_per_action = {0: [], 1: [], 2: []}
N_STEPS = 300

for step in range(N_STEPS):
    # Force each action in rotation
    action = step % 3
    obs, reward, done, info = vec_env.step(np.array([action]))
    reward_per_action[action].append(reward[0])

    if done[0]:
        obs = vec_env.reset()

print(f"\n   Rewards moyens par action (sur {N_STEPS//3} essais chaque):")
print(f"")
names = {0: "SELL", 1: "HOLD", 2: "BUY "}

for a, rewards in reward_per_action.items():
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    count = len(rewards)

    # Visual
    if mean_r >= 0:
        bar = "+" * min(int(mean_r * 1000), 20)
    else:
        bar = "-" * min(int(-mean_r * 1000), 20)

    print(f"   {names[a]}: mean={mean_r:+.6f} | std={std_r:.6f} | n={count}")

# Compare
hold_reward = np.mean(reward_per_action[1])
trade_reward = np.mean(reward_per_action[0] + reward_per_action[2])

print(f"\n   Comparaison:")
print(f"   - HOLD moyen:  {hold_reward:+.6f}")
print(f"   - TRADE moyen: {trade_reward:+.6f}")

if hold_reward > trade_reward:
    print(f"\n   🚨 HOLD > TRADE: L'agent est RÉCOMPENSÉ pour ne rien faire!")
    print(f"   → C'est la cause principale du non-trading")
else:
    print(f"\n   ✅ TRADE >= HOLD: Le reward favorise le trading")

# ============================================================================
# TEST F: SOLUTIONS SELON DIAGNOSTIC
# ============================================================================
print("\n" + "="*70)
print("SOLUTIONS SELON DIAGNOSTIC")
print("="*70)

print(f"""
   RÉSUMÉ DES PROBLÈMES DÉTECTÉS:
   ─────────────────────────────
   - Distribution: {hold_pct:.0f}% HOLD {'(PROBLÈME!)' if hold_pct > 80 else '(OK)'}
   - Entropy: {ent_coef if model else 'N/A'}
   - Random vs Trained: {'≈ égaux (PROBLÈME!)' if model and abs(diff) < 0.01 else 'OK' if model else 'N/A'}
   - Reward HOLD vs TRADE: {'HOLD > TRADE (PROBLÈME!)' if hold_reward > trade_reward else 'OK'}

   SOLUTIONS À APPLIQUER:
   ─────────────────────
""")

solutions = []

# Solution 1: Si HOLD domine
if hold_pct > 80:
    solutions.append("""
   1️⃣ SI L'AGENT A PEUR D'AGIR:
      # Ajouter un coût à l'inaction (déjà fait en V2)
      if action == HOLD and has_no_position:
          reward -= 0.02  # Pénalité pour inaction prolongée

      # Dans trading_env_v2_ultimate.py (DÉJÀ IMPLÉMENTÉ):
      # → -0.02 si consecutive_holds > 30
""")

# Solution 2: Si pas assez d'exploration
if model and ent_coef < 0.10:
    solutions.append("""
   2️⃣ SI PAS ASSEZ D'EXPLORATION:
      # Pour PPO, augmenter ent_coef (déjà fait en V2)
      ent_coef = 0.25  # Au lieu de 0.05
      ent_coef_min = 0.12  # Ne pas descendre trop bas

      # Dans train_CRITIC_BOOST_LSTM.py (DÉJÀ IMPLÉMENTÉ):
      # → Entropy 0-50%: 0.25
      # → Entropy minimum: 0.12
""")

# Solution 3: Si reward sparse
if hold_reward > trade_reward:
    solutions.append("""
   3️⃣ SI REWARD SPARSE (que à la fin):
      # Ajouter des rewards intermédiaires
      # - Reward pour unrealized PnL
      # - Reward pour "bonne direction" même sans clôture
      # - Bonus pour chaque action BUY/SELL

      # Dans trading_env_v2_ultimate.py (DÉJÀ IMPLÉMENTÉ):
      # → +0.03 pour chaque BUY/SELL
      # → -0.03 si >90% single action
""")

if solutions:
    for s in solutions:
        print(s)
else:
    print(f"   ✅ Pas de problème majeur détecté!")

print(f"""
   ═══════════════════════════════════════════════════════════════════
   PROCHAINE ÉTAPE:

   Les fixes sont déjà implémentés dans la V2!
   → Relancer le training avec les nouveaux paramètres:

   cd "C:\\Users\\lbye3\\Desktop\\GoldRL\\AGENT\\AGENT 7\\ENTRAINEMENT\\FICHIER IMPORTANT AGENT 7\\launchers"
   LAUNCH_TRAINING_500K.bat
   ═══════════════════════════════════════════════════════════════════
""")

print("="*70)
print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

vec_env.close()
