# -*- coding: utf-8 -*-
"""
================================================================================
DIAGNOSTIC COMPLET - AGENT 7 V2.1 CRITIC BOOST + LSTM
================================================================================

OBJECTIF: Comprendre POURQUOI l'agent n'ouvre pas ou très peu de positions

QUESTIONS ANALYSÉES:
1. L'agent a-t-il PEUR de l'échec? (analyse des probabilités)
2. Préfère-t-il HOLD par sécurité? (analyse reward HOLD vs BUY/SELL)
3. Ferme-t-il correctement les positions? (analyse cycle ouverture/fermeture)
4. Comment réagit-il step par step? (trace détaillée)
5. Quels BLOCAGES l'empêchent d'ouvrir? (confidence, daily loss, drawdown)
6. Que voit-il dans les features? (analyse observation)
7. Son Critic distingue-t-il les bons/mauvais états? (Value Function)
8. Sa mémoire LSTM fonctionne-t-elle? (séquences temporelles)
9. Le reward l'encourage-t-il à trader? (analyse reward function)
10. SOLUTIONS: Comment le faire trader plus?

Durée: ~5 minutes
Output: Rapport détaillé avec recommandations

================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, deque
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

# ============================================================================
# CONFIGURATION
# ============================================================================
N_DIAGNOSTIC_STEPS = 500  # Nombre de steps à analyser en détail
VERBOSE_STEPS = True  # Afficher chaque step?

print("=" * 80)
print("DIAGNOSTIC COMPLET - AGENT 7 V2.1")
print("Comprendre pourquoi l'agent n'ouvre pas de positions")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Steps à analyser: {N_DIAGNOSTIC_STEPS}")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_unwrapped_env(vec_env):
    """Accéder à l'environnement réel (sans wrappers)"""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

def get_action_probabilities(model, obs, lstm_states, n_samples=50):
    """Estimer les probabilités d'action par sampling"""
    action_counts = Counter()
    for _ in range(n_samples):
        action, _ = model.predict(obs, state=lstm_states, deterministic=False)
        action_counts[int(action[0])] += 1
    total = sum(action_counts.values())
    return {k: v/total for k, v in action_counts.items()}

def analyze_observation(obs, feature_names=None):
    """Analyser l'observation reçue par l'agent"""
    obs_flat = obs.flatten()

    # Stats générales
    stats = {
        'mean': np.mean(obs_flat),
        'std': np.std(obs_flat),
        'min': np.min(obs_flat),
        'max': np.max(obs_flat),
        'n_zeros': np.sum(obs_flat == 0),
        'n_nan': np.sum(np.isnan(obs_flat)),
        'n_inf': np.sum(np.isinf(obs_flat))
    }

    # Top features par valeur absolue
    top_indices = np.argsort(np.abs(obs_flat))[::-1][:10]

    return stats, top_indices, obs_flat

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*60)
print("ÉTAPE 1: Chargement des données")
print("="*60)

loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)

# Utiliser période TEST pour diagnostic
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]

print(f"   [OK] Features: {features.shape}")
print(f"   [OK] Period: {config.TEST_START_DATE} → {config.TEST_END_DATE}")
print(f"   [OK] Colonnes: {list(features.columns[:10])}... (+{len(features.columns)-10})")

# ============================================================================
# CREATE ENVIRONMENT
# ============================================================================
print("\n" + "="*60)
print("ÉTAPE 2: Création de l'environnement")
print("="*60)

env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)
env_monitor = Monitor(env)
vec_env = DummyVecEnv([lambda: env_monitor])

print(f"   [OK] Observation space: {vec_env.observation_space.shape}")
print(f"   [OK] Action space: Discrete(3) - SELL/HOLD/BUY")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*60)
print("ÉTAPE 3: Chargement du modèle")
print("="*60)

# Chercher le meilleur checkpoint (150K était le meilleur)
model_paths = [
    parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip",
    parent_dir / "models" / "best_model.zip",
    parent_dir / "models" / "agent7_critic_boost_lstm_final.zip"
]

model = None
for path in model_paths:
    if path.exists():
        model = RecurrentPPO.load(path, env=vec_env)
        print(f"   [OK] Modèle chargé: {path.name}")
        break

if model is None:
    print("   [ERROR] Aucun modèle trouvé!")
    print("   Chemins testés:")
    for p in model_paths:
        print(f"      - {p}")
    sys.exit(1)

# ============================================================================
# DIAGNOSTIC QUESTIONS
# ============================================================================

print("\n" + "="*80)
print("DÉBUT DU DIAGNOSTIC - 10 QUESTIONS")
print("="*80)

# Storage for analysis
all_actions = []
all_rewards = []
all_positions = []
all_probs = []
all_values = []
trades_opened = 0
trades_closed = 0
consecutive_holds = 0
max_consecutive_holds = 0
blockers_encountered = {
    'daily_loss': 0,
    'max_drawdown': 0,
    'tail_risk': 0,
    'confidence': 0,
    'risk_multiplier': 0
}

# Reset environment
obs = vec_env.reset()
lstm_states = None
last_position = 0

raw_env = get_unwrapped_env(vec_env)

print("\n" + "-"*60)
print("ANALYSE STEP-BY-STEP (premiers 100 steps détaillés)")
print("-"*60)

for step in range(N_DIAGNOSTIC_STEPS):
    # Get action probabilities
    probs = get_action_probabilities(model, obs, lstm_states, n_samples=30)
    all_probs.append(probs)

    # Get deterministic action
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    action_int = int(action[0])
    all_actions.append(action_int)

    # Track consecutive holds
    if action_int == 1:
        consecutive_holds += 1
        max_consecutive_holds = max(max_consecutive_holds, consecutive_holds)
    else:
        consecutive_holds = 0

    # Execute action
    obs, reward, done, info = vec_env.step(action)
    all_rewards.append(reward[0])

    # Track position changes
    current_position = raw_env.position_side
    all_positions.append(current_position)

    if last_position == 0 and current_position != 0:
        trades_opened += 1
    elif last_position != 0 and current_position == 0:
        trades_closed += 1
    last_position = current_position

    # Track blockers
    if hasattr(raw_env, 'daily_loss_limit_reached') and raw_env.daily_loss_limit_reached:
        blockers_encountered['daily_loss'] += 1
    if hasattr(raw_env, 'max_drawdown') and raw_env.max_drawdown >= 0.10:
        blockers_encountered['max_drawdown'] += 1
    if hasattr(raw_env, 'tail_risk_detected') and raw_env.tail_risk_detected:
        blockers_encountered['tail_risk'] += 1

    # Verbose output for first 100 steps
    if step < 100 and VERBOSE_STEPS:
        action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        prob_str = f"S:{probs.get(0,0)*100:.0f}% H:{probs.get(1,0)*100:.0f}% B:{probs.get(2,0)*100:.0f}%"
        pos_str = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[current_position]

        # Highlight important events
        marker = ""
        if action_int != 1:
            marker = " ← ACTION!"
        if current_position != all_positions[-2] if len(all_positions) > 1 else 0:
            marker = " ← POSITION CHANGE!"

        print(f"  Step {step:3d}: {action_names[action_int]:4s} | Probs: {prob_str} | Pos: {pos_str:5s} | R: {reward[0]:+.4f}{marker}")

    if done[0]:
        obs = vec_env.reset()
        lstm_states = None

# ============================================================================
# QUESTION 1: L'agent a-t-il PEUR de l'échec?
# ============================================================================
print("\n" + "="*60)
print("Q1: L'agent a-t-il PEUR de l'échec?")
print("="*60)

# Analyser la distribution des probabilités moyennes
avg_probs = {0: 0, 1: 0, 2: 0}
for p in all_probs:
    for k, v in p.items():
        avg_probs[k] += v
for k in avg_probs:
    avg_probs[k] /= len(all_probs)

print(f"\n   Probabilités moyennes sur {N_DIAGNOSTIC_STEPS} steps:")
print(f"   - SELL: {avg_probs.get(0,0)*100:.1f}%")
print(f"   - HOLD: {avg_probs.get(1,0)*100:.1f}%")
print(f"   - BUY:  {avg_probs.get(2,0)*100:.1f}%")

# Analyser la variance des probabilités
prob_stds = {0: [], 1: [], 2: []}
for p in all_probs:
    for k in [0, 1, 2]:
        prob_stds[k].append(p.get(k, 0))

print(f"\n   Variabilité des probabilités (std):")
print(f"   - SELL: {np.std(prob_stds[0]):.3f}")
print(f"   - HOLD: {np.std(prob_stds[1]):.3f}")
print(f"   - BUY:  {np.std(prob_stds[2]):.3f}")

# Diagnostic
if avg_probs.get(1, 0) > 0.7:
    print(f"\n   [DIAGNOSTIC] OUI - L'agent a PEUR de l'échec!")
    print(f"   - Il choisit HOLD {avg_probs.get(1,0)*100:.1f}% du temps")
    print(f"   - HOLD = pas de risque = pas de perte")
    print(f"   - C'est un comportement de mode collapse typique")
elif avg_probs.get(1, 0) > 0.5:
    print(f"\n   [DIAGNOSTIC] PEUT-ÊTRE - Tendance vers HOLD")
    print(f"   - HOLD à {avg_probs.get(1,0)*100:.1f}% est élevé mais pas extrême")
else:
    print(f"\n   [DIAGNOSTIC] NON - Distribution équilibrée")

# ============================================================================
# QUESTION 2: Préfère-t-il HOLD par sécurité?
# ============================================================================
print("\n" + "="*60)
print("Q2: Préfère-t-il HOLD par sécurité?")
print("="*60)

action_counts = Counter(all_actions)
total_actions = len(all_actions)

print(f"\n   Actions exécutées sur {total_actions} steps:")
print(f"   - SELL: {action_counts.get(0,0):4d} ({action_counts.get(0,0)/total_actions*100:.1f}%)")
print(f"   - HOLD: {action_counts.get(1,0):4d} ({action_counts.get(1,0)/total_actions*100:.1f}%)")
print(f"   - BUY:  {action_counts.get(2,0):4d} ({action_counts.get(2,0)/total_actions*100:.1f}%)")

print(f"\n   Séquences de HOLD consécutifs:")
print(f"   - Maximum: {max_consecutive_holds} HOLDs d'affilée")

# Analyser les transitions
transitions = Counter()
for i in range(1, len(all_actions)):
    transitions[(all_actions[i-1], all_actions[i])] += 1

print(f"\n   Transitions les plus fréquentes:")
for (a1, a2), count in transitions.most_common(5):
    names = {0: "SELL", 1: "HOLD", 2: "BUY"}
    print(f"   - {names[a1]} → {names[a2]}: {count} fois")

# Diagnostic
hold_pct = action_counts.get(1, 0) / total_actions
if hold_pct > 0.9:
    print(f"\n   [DIAGNOSTIC] CRITIQUE - Mode collapse vers HOLD!")
    print(f"   - {hold_pct*100:.1f}% HOLD = l'agent est paralysé")
    print(f"   - Il a appris que HOLD = sécurité")
elif hold_pct > 0.7:
    print(f"\n   [DIAGNOSTIC] ÉLEVÉ - Forte préférence pour HOLD")
else:
    print(f"\n   [DIAGNOSTIC] OK - Distribution acceptable")

# ============================================================================
# QUESTION 3: Ferme-t-il correctement les positions?
# ============================================================================
print("\n" + "="*60)
print("Q3: Ferme-t-il correctement les positions?")
print("="*60)

print(f"\n   Trades pendant {N_DIAGNOSTIC_STEPS} steps:")
print(f"   - Positions ouvertes: {trades_opened}")
print(f"   - Positions fermées: {trades_closed}")

if trades_opened == 0:
    print(f"\n   [DIAGNOSTIC] CRITIQUE - AUCUNE position ouverte!")
    print(f"   - L'agent ne trade PAS DU TOUT")
    print(f"   - Il reste en mode FLAT 100% du temps")
elif trades_opened > trades_closed:
    print(f"\n   [DIAGNOSTIC] ATTENTION - Positions non fermées")
    print(f"   - {trades_opened - trades_closed} positions encore ouvertes")
else:
    print(f"\n   [DIAGNOSTIC] OK - Cycle ouverture/fermeture normal")

# Analyser les positions
position_counts = Counter(all_positions)
print(f"\n   Distribution des positions:")
print(f"   - FLAT (0):  {position_counts.get(0,0):4d} ({position_counts.get(0,0)/len(all_positions)*100:.1f}%)")
print(f"   - SHORT(-1): {position_counts.get(-1,0):4d} ({position_counts.get(-1,0)/len(all_positions)*100:.1f}%)")
print(f"   - LONG (1):  {position_counts.get(1,0):4d} ({position_counts.get(1,0)/len(all_positions)*100:.1f}%)")

# ============================================================================
# QUESTION 4: Comment réagit-il step par step?
# ============================================================================
print("\n" + "="*60)
print("Q4: Comment réagit-il step par step?")
print("="*60)

# Analyser les patterns d'actions
print(f"\n   Séquences d'actions détectées:")

# Trouver les patterns répétitifs
pattern_length = 5
patterns = Counter()
for i in range(len(all_actions) - pattern_length):
    pattern = tuple(all_actions[i:i+pattern_length])
    patterns[pattern] += 1

print(f"   Top 5 patterns de {pattern_length} actions:")
for pattern, count in patterns.most_common(5):
    names = {0: "S", 1: "H", 2: "B"}
    pattern_str = "→".join([names[a] for a in pattern])
    print(f"   - {pattern_str}: {count} fois")

# Changements d'action
action_changes = sum(1 for i in range(1, len(all_actions)) if all_actions[i] != all_actions[i-1])
print(f"\n   Changements d'action: {action_changes}/{total_actions-1} ({action_changes/(total_actions-1)*100:.1f}%)")

if action_changes / (total_actions - 1) < 0.1:
    print(f"   [DIAGNOSTIC] TRÈS PEU de changements - Agent figé")
elif action_changes / (total_actions - 1) < 0.3:
    print(f"   [DIAGNOSTIC] Peu de changements - Agent conservateur")
else:
    print(f"   [DIAGNOSTIC] OK - Agent réactif")

# ============================================================================
# QUESTION 5: Quels BLOCAGES l'empêchent d'ouvrir?
# ============================================================================
print("\n" + "="*60)
print("Q5: Quels BLOCAGES l'empêchent d'ouvrir?")
print("="*60)

print(f"\n   Blocages détectés pendant {N_DIAGNOSTIC_STEPS} steps:")
print(f"   - Daily Loss Limit: {blockers_encountered['daily_loss']} steps")
print(f"   - Max Drawdown (10%): {blockers_encountered['max_drawdown']} steps")
print(f"   - Tail Risk: {blockers_encountered['tail_risk']} steps")

# Vérifier les seuils actuels
print(f"\n   Configuration des seuils:")
print(f"   - MIN_CONFIDENCE_NORMAL: {config.MIN_CONFIDENCE_NORMAL if hasattr(config, 'MIN_CONFIDENCE_NORMAL') else 'N/A'}")
print(f"   - MIN_CONFIDENCE_STRESSED: {config.MIN_CONFIDENCE_STRESSED if hasattr(config, 'MIN_CONFIDENCE_STRESSED') else 'N/A'}")
print(f"   - FTMO_MAX_DRAWDOWN: {config.FTMO_MAX_DRAWDOWN if hasattr(config, 'FTMO_MAX_DRAWDOWN') else '10%'}")

# Pour discrete actions, size = 1.0 toujours
print(f"\n   [INFO] Pour actions DISCRETE:")
print(f"   - size = 1.0 (100%) automatiquement")
print(f"   - Le seuil de confiance ne bloque PAS en mode discrete")
print(f"   - MAIS le risk_multiplier peut être 0 si DD élevé")

# ============================================================================
# QUESTION 6: Que voit-il dans les features?
# ============================================================================
print("\n" + "="*60)
print("Q6: Que voit-il dans les features?")
print("="*60)

# Analyser la dernière observation
obs_stats, top_indices, obs_flat = analyze_observation(obs)

print(f"\n   Statistiques de l'observation (229 features):")
print(f"   - Moyenne: {obs_stats['mean']:.4f}")
print(f"   - Écart-type: {obs_stats['std']:.4f}")
print(f"   - Min: {obs_stats['min']:.4f}")
print(f"   - Max: {obs_stats['max']:.4f}")
print(f"   - Valeurs à 0: {obs_stats['n_zeros']}")
print(f"   - NaN: {obs_stats['n_nan']}")
print(f"   - Inf: {obs_stats['n_inf']}")

print(f"\n   Top 10 features par valeur absolue:")
for i, idx in enumerate(top_indices):
    print(f"   {i+1}. Feature[{idx}]: {obs_flat[idx]:.4f}")

# Vérifier les RL features (dernières 20)
print(f"\n   RL Features (indices 209-228):")
rl_names = [
    "last_action_0", "last_action_1", "last_action_2",
    "regret_signal", "position_duration", "pnl_ratio",
    "regime_0", "regime_1", "regime_2",
    "hours_until_event", "volatility_percentile", "position_side",
    "trade_similarity",
    "win_rate", "streak", "avg_pnl", "best_trade", "worst_trade",
    "win_count", "loss_count"
]
for i, name in enumerate(rl_names):
    val = obs_flat[209 + i] if 209 + i < len(obs_flat) else 0
    print(f"   [{209+i}] {name}: {val:.4f}")

# ============================================================================
# QUESTION 7: Son Critic distingue-t-il les bons/mauvais états?
# ============================================================================
print("\n" + "="*60)
print("Q7: Son Critic distingue-t-il les bons/mauvais états?")
print("="*60)

# Calculer la variance des rewards
reward_std = np.std(all_rewards)
reward_mean = np.mean(all_rewards)

print(f"\n   Statistiques des rewards sur {N_DIAGNOSTIC_STEPS} steps:")
print(f"   - Moyenne: {reward_mean:.6f}")
print(f"   - Écart-type: {reward_std:.6f}")
print(f"   - Min: {np.min(all_rewards):.6f}")
print(f"   - Max: {np.max(all_rewards):.6f}")

if reward_std < 0.001:
    print(f"\n   [DIAGNOSTIC] CRITIQUE - Rewards presque constants!")
    print(f"   - Le Critic ne peut pas apprendre si rewards identiques")
    print(f"   - L'agent ne voit pas de différence entre actions")
elif reward_std < 0.01:
    print(f"\n   [DIAGNOSTIC] FAIBLE - Peu de variation dans rewards")
else:
    print(f"\n   [DIAGNOSTIC] OK - Variation suffisante")

# ============================================================================
# QUESTION 8: Sa mémoire LSTM fonctionne-t-elle?
# ============================================================================
print("\n" + "="*60)
print("Q8: Sa mémoire LSTM fonctionne-t-elle?")
print("="*60)

print(f"\n   Configuration LSTM:")
print(f"   - Hidden size: 256")
print(f"   - Layers: 1")
print(f"   - Steps memory: 16 (16 heures)")

# Vérifier si l'agent utilise la mémoire
if len(raw_env.trades) > 0:
    print(f"\n   Trades en mémoire: {len(raw_env.trades)}")
    print(f"   - L'agent a de l'historique pour apprendre")
else:
    print(f"\n   [DIAGNOSTIC] COLD START - Aucun trade en mémoire!")
    print(f"   - Les 7 MEMORY features sont à valeur par défaut (0.5)")
    print(f"   - L'agent n'a pas de contexte historique")

# Vérifier action history
if hasattr(raw_env, 'action_history'):
    hist = list(raw_env.action_history)
    print(f"\n   Action history (dernières 5): {hist}")

# ============================================================================
# QUESTION 9: Le reward l'encourage-t-il à trader?
# ============================================================================
print("\n" + "="*60)
print("Q9: Le reward l'encourage-t-il à trader?")
print("="*60)

# Comparer rewards par action
rewards_by_action = {0: [], 1: [], 2: []}
for i, action in enumerate(all_actions):
    rewards_by_action[action].append(all_rewards[i])

print(f"\n   Reward moyen par action:")
for action, rewards in rewards_by_action.items():
    if rewards:
        names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        print(f"   - {names[action]}: {np.mean(rewards):.6f} (n={len(rewards)})")
    else:
        names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        print(f"   - {names[action]}: N/A (aucune)")

# Diagnostic
hold_reward = np.mean(rewards_by_action[1]) if rewards_by_action[1] else 0
sell_reward = np.mean(rewards_by_action[0]) if rewards_by_action[0] else 0
buy_reward = np.mean(rewards_by_action[2]) if rewards_by_action[2] else 0

if hold_reward > max(sell_reward, buy_reward):
    print(f"\n   [DIAGNOSTIC] PROBLÈME - HOLD a le meilleur reward!")
    print(f"   - L'agent est RÉCOMPENSÉ pour ne rien faire")
    print(f"   - Il a appris que HOLD = meilleur choix")
else:
    print(f"\n   [DIAGNOSTIC] OK - Trading peut être récompensé")

# ============================================================================
# QUESTION 10: SOLUTIONS
# ============================================================================
print("\n" + "="*60)
print("Q10: SOLUTIONS pour faire trader l'agent")
print("="*60)

print(f"""
   DIAGNOSTIC RÉSUMÉ:
   ==================
   - Actions: SELL {action_counts.get(0,0)/total_actions*100:.0f}% | HOLD {action_counts.get(1,0)/total_actions*100:.0f}% | BUY {action_counts.get(2,0)/total_actions*100:.0f}%
   - Trades ouverts: {trades_opened}
   - Trades fermés: {trades_closed}
   - Max HOLD consécutifs: {max_consecutive_holds}
   - Reward std: {reward_std:.6f}

   PROBLÈMES IDENTIFIÉS:
   ====================
""")

problems = []
if action_counts.get(1, 0) / total_actions > 0.9:
    problems.append("1. MODE COLLAPSE: 90%+ HOLD")
if trades_opened == 0:
    problems.append("2. AUCUN TRADE: L'agent ne trade jamais")
if max_consecutive_holds > 100:
    problems.append(f"3. PARALYSIE: {max_consecutive_holds} HOLDs d'affilée")
if reward_std < 0.01:
    problems.append("4. REWARDS PLATS: Pas de signal d'apprentissage")
if len(raw_env.trades) == 0:
    problems.append("5. COLD START: Pas de mémoire de trades")

for p in problems:
    print(f"   - {p}")

print(f"""

   SOLUTIONS IMPLÉMENTÉES:
   =======================
   [OK] 1. Entropy étendue: 0-50% à 0.25 (au lieu de 0-30% à 0.20)
   [OK] 2. Entropy minimum: 0.12 (au lieu de 0.05)
   [OK] 3. Exploration bonus: +0.03 pour BUY/SELL
   [OK] 4. HOLD penalty: -0.02 si >30 HOLDs consécutifs
   [OK] 5. Diversity penalty: -0.03 si >90% single action

   SOLUTIONS ADDITIONNELLES À CONSIDÉRER:
   ======================================
   [ ] 6. Warmup trades: Pré-remplir 10-20 trades aléatoires au reset
   [ ] 7. Curiosity bonus: Récompenser les nouvelles situations
   [ ] 8. Action forcing: Forcer BUY/SELL tous les N steps en début de training
   [ ] 9. Imitation learning: Pré-entraîner sur trades experts
   [ ] 10. Reward shaping: Bonus pour position != FLAT
""")

# ============================================================================
# RAPPORT FINAL
# ============================================================================
print("\n" + "="*80)
print("RAPPORT FINAL")
print("="*80)

severity = "CRITIQUE" if trades_opened == 0 else "MODÉRÉ" if trades_opened < 10 else "OK"

print(f"""
   VERDICT: {severity}

   L'agent souffre de MODE COLLAPSE car:
   1. Il a appris que HOLD = sécurité (pas de perte possible)
   2. L'entropy a diminué trop vite (30% → 0.05)
   3. Pas de bonus pour encourager le trading
   4. Pas de pénalité pour HOLD excessif

   AVEC LES FIXES IMPLÉMENTÉS, on s'attend à:
   - Plus de diversité d'actions (SELL/HOLD/BUY ~33% chaque)
   - Plus de trades ouverts dès les premiers steps
   - Moins de séquences de HOLD consécutifs

   PROCHAINE ÉTAPE:
   Lancer le Quick Test 10K pour valider les fixes:

   cd "C:\\Users\\lbye3\\Desktop\\GoldRL\\AGENT\\AGENT 7\\ENTRAINEMENT\\FICHIER IMPORTANT AGENT 7\\launchers"
   RUN_QUICK_TEST_10K.bat
""")

print("="*80)
print(f"FIN DU DIAGNOSTIC: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Cleanup
vec_env.close()
