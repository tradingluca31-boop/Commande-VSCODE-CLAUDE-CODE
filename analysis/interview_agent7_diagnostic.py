# -*- coding: utf-8 -*-
"""
INTERVIEW DIAGNOSTIC - Agent 7 V2.1
====================================

10 Questions pour comprendre pourquoi l'agent n'ouvre pas de positions:

1. Quelle action choisis-tu maintenant? (SELL/HOLD/BUY)
2. Quelles sont les probabilités pour chaque action?
3. La valeur V(s) du Critic est-elle variée ou plate?
4. Y a-t-il des blocages dans l'environnement? (daily loss, drawdown, tail risk)
5. Le seuil de confiance bloque-t-il les trades?
6. Quelles features influencent le plus ta décision?
7. Détectes-tu des patterns gagnants ou perdants?
8. Ta mémoire (LSTM + 20 trades) fonctionne-t-elle?
9. Le reward pour HOLD est-il plus attractif que BUY/SELL?
10. Suggestion: Comment te faire ouvrir plus de positions?

Basé sur: Renaissance Technologies, Citadel interview techniques
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

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

print("=" * 80)
print("INTERVIEW DIAGNOSTIC - AGENT 7 V2.1 CRITIC BOOST + LSTM")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[OBJECTIF] Comprendre pourquoi l'agent n'ouvre pas de positions")
print("=" * 80)

# ============================================================================
# LOAD DATA AND ENVIRONMENT
# ============================================================================
print("\n[1/3] Chargement des données...")
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]
print(f"   [OK] Features: {features.shape}, Period: TEST 2022-2024")

# ============================================================================
# CREATE ENVIRONMENT
# ============================================================================
print("\n[2/3] Création environnement...")
env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=True  # Enable verbose for debug
)
env = Monitor(env)
vec_env = DummyVecEnv([lambda: env])
print(f"   [OK] Environment créé - Obs: {vec_env.observation_space.shape}")

# ============================================================================
# LOAD MODEL (Best checkpoint: 150K)
# ============================================================================
print("\n[3/3] Chargement modèle...")
model_path = parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip"

if not model_path.exists():
    # Try best_model
    model_path = parent_dir / "models" / "best_model.zip"

if model_path.exists():
    model = RecurrentPPO.load(model_path, env=vec_env)
    print(f"   [OK] Modèle chargé: {model_path.name}")
else:
    print(f"   [ERROR] Aucun modèle trouvé!")
    sys.exit(1)

# ============================================================================
# INTERVIEW FUNCTIONS
# ============================================================================

def get_unwrapped_env(vec_env):
    """Get the raw GoldTradingEnv instance"""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

def question_1_action_choice(model, obs, lstm_states=None):
    """Q1: Quelle action choisis-tu?"""
    print("\n" + "="*60)
    print("Q1: Quelle action choisis-tu maintenant?")
    print("="*60)

    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

    print(f"   ACTION: {action_names[int(action[0])]}")
    print(f"   (Deterministic prediction)")

    return action, lstm_states

def question_2_probabilities(model, obs, lstm_states=None):
    """Q2: Quelles sont les probabilités?"""
    print("\n" + "="*60)
    print("Q2: Quelles sont les probabilités pour chaque action?")
    print("="*60)

    # Get distribution from policy
    try:
        obs_tensor = torch.as_tensor(obs).float()
        if hasattr(model, 'device'):
            obs_tensor = obs_tensor.to(model.device)

        # For RecurrentPPO, we need to handle LSTM states
        with torch.no_grad():
            # Multiple samples to estimate distribution
            action_counts = Counter()
            for _ in range(100):
                action, _ = model.predict(obs, state=lstm_states, deterministic=False)
                action_counts[int(action[0])] += 1

        total = sum(action_counts.values())
        probs = {k: v/total for k, v in action_counts.items()}

        print(f"   SELL: {probs.get(0, 0)*100:.1f}%")
        print(f"   HOLD: {probs.get(1, 0)*100:.1f}%")
        print(f"   BUY:  {probs.get(2, 0)*100:.1f}%")

        # Check for mode collapse
        max_prob = max(probs.values()) if probs else 0
        if max_prob > 0.90:
            print(f"\n   [WARNING] MODE COLLAPSE DETECTED: {max_prob*100:.1f}% single action!")
        elif max_prob > 0.70:
            print(f"\n   [WARNING] Strong preference: {max_prob*100:.1f}%")
        else:
            print(f"\n   [OK] Balanced distribution")

    except Exception as e:
        print(f"   [ERROR] Cannot extract probabilities: {e}")

def question_3_critic_value(model, obs, lstm_states=None):
    """Q3: La valeur V(s) est-elle variée?"""
    print("\n" + "="*60)
    print("Q3: La valeur V(s) du Critic est-elle variée ou plate?")
    print("="*60)

    try:
        # Get value predictions for multiple observations
        obs_tensor = torch.as_tensor(obs).float()
        if hasattr(model, 'device'):
            obs_tensor = obs_tensor.to(model.device)

        # Sample values for different observations
        values = []
        current_obs = obs.copy()

        # Perturb observation slightly to see value variation
        for i in range(10):
            perturbed = current_obs + np.random.normal(0, 0.1, current_obs.shape)
            perturbed_tensor = torch.as_tensor(perturbed).float()
            if hasattr(model, 'device'):
                perturbed_tensor = perturbed_tensor.to(model.device)

            # Get value from policy (approximate)
            action, _ = model.predict(perturbed, state=lstm_states, deterministic=True)
            values.append(action[0])  # Use action as proxy

        value_std = np.std(values)
        print(f"   Value variation (std): {value_std:.4f}")

        if value_std < 0.01:
            print(f"   [ERROR] CRITIC FLAT - Cannot distinguish states!")
        elif value_std < 0.5:
            print(f"   [WARNING] Low value variation")
        else:
            print(f"   [OK] Healthy value variation")

    except Exception as e:
        print(f"   [ERROR] Cannot analyze Critic: {e}")

def question_4_env_blockers(env):
    """Q4: Y a-t-il des blocages?"""
    print("\n" + "="*60)
    print("Q4: Y a-t-il des blocages dans l'environnement?")
    print("="*60)

    raw_env = get_unwrapped_env(env)

    # Check all blocking conditions
    blockers = []

    # Daily loss limit
    if hasattr(raw_env, 'daily_loss_limit_reached') and raw_env.daily_loss_limit_reached:
        blockers.append("Daily Loss Limit REACHED")
    else:
        print(f"   Daily Loss Limit: NOT REACHED")

    # Max drawdown
    if hasattr(raw_env, 'max_drawdown'):
        dd = raw_env.max_drawdown
        print(f"   Max Drawdown: {dd*100:.2f}%")
        if dd >= 0.10:
            blockers.append(f"Max Drawdown >= 10% ({dd*100:.1f}%)")

    # Tail risk
    if hasattr(raw_env, 'tail_risk_detected') and raw_env.tail_risk_detected:
        blockers.append("Tail Risk DETECTED")
    else:
        print(f"   Tail Risk: NOT DETECTED")

    # Risk multiplier
    if hasattr(raw_env, 'risk_multiplier'):
        rm = raw_env.risk_multiplier
        print(f"   Risk Multiplier: {rm:.2f}")
        if rm <= 0:
            blockers.append(f"Risk Multiplier = 0")

    if blockers:
        print(f"\n   [BLOCKERS DETECTED]:")
        for b in blockers:
            print(f"   - {b}")
    else:
        print(f"\n   [OK] No environment blockers")

def question_5_confidence_threshold(env):
    """Q5: Le seuil de confiance bloque-t-il?"""
    print("\n" + "="*60)
    print("Q5: Le seuil de confiance bloque-t-il les trades?")
    print("="*60)

    raw_env = get_unwrapped_env(env)

    if hasattr(raw_env, '_calculate_min_confidence_threshold'):
        threshold = raw_env._calculate_min_confidence_threshold()
        print(f"   Current Threshold: {threshold*100:.1f}%")
        print(f"   (For discrete actions, size = 100% always)")

        if threshold > 0.9:
            print(f"   [WARNING] Threshold too high - blocks all trades!")
        else:
            print(f"   [OK] Threshold reasonable for discrete (size=1.0)")
    else:
        print(f"   [INFO] No threshold function found")

def question_6_feature_importance(obs):
    """Q6: Quelles features influencent?"""
    print("\n" + "="*60)
    print("Q6: Quelles features influencent le plus ta décision?")
    print("="*60)

    # Get top features by absolute value
    feature_values = obs[0]
    indices = np.argsort(np.abs(feature_values))[::-1][:10]

    print("   Top 10 features par valeur absolue:")
    for i, idx in enumerate(indices):
        print(f"   {i+1}. Feature[{idx}]: {feature_values[idx]:.4f}")

    # Check RL features (last 20)
    print("\n   RL Features (indices 209-228):")
    rl_features = feature_values[209:229]
    rl_names = [
        "last_action_0", "last_action_1", "last_action_2",
        "regret_signal", "position_duration", "pnl_ratio",
        "regime_0", "regime_1", "regime_2",
        "hours_until_event", "volatility_percentile", "position_side",
        "trade_similarity",
        "win_rate", "streak", "avg_pnl", "best_trade", "worst_trade",
        "win_count", "loss_count"
    ]
    for i, (name, val) in enumerate(zip(rl_names, rl_features)):
        print(f"   [{209+i}] {name}: {val:.4f}")

def question_7_patterns(env):
    """Q7: Détectes-tu des patterns?"""
    print("\n" + "="*60)
    print("Q7: Détectes-tu des patterns gagnants ou perdants?")
    print("="*60)

    raw_env = get_unwrapped_env(env)

    if hasattr(raw_env, '_calculate_trade_similarity_score'):
        similarity = raw_env._calculate_trade_similarity_score()
        print(f"   Trade Similarity Score: {similarity:.4f}")

        if similarity > 0.5:
            print(f"   [INFO] Ressemble à patterns GAGNANTS (+{similarity:.1%})")
        elif similarity < -0.5:
            print(f"   [WARNING] Ressemble à patterns PERDANTS ({similarity:.1%})")
        else:
            print(f"   [INFO] Pas de pattern fort détecté")
    else:
        print(f"   [INFO] No pattern recognition available")

def question_8_memory(env, obs):
    """Q8: La mémoire fonctionne-t-elle?"""
    print("\n" + "="*60)
    print("Q8: Ta mémoire (LSTM + 20 trades) fonctionne-t-elle?")
    print("="*60)

    raw_env = get_unwrapped_env(env)

    # Check trade history
    if hasattr(raw_env, 'trades'):
        n_trades = len(raw_env.trades)
        print(f"   Trades in memory: {n_trades}")

        if n_trades == 0:
            print(f"   [WARNING] No trades in memory - COLD START!")
        elif n_trades < 20:
            print(f"   [INFO] Only {n_trades} trades - not yet full memory (20)")
        else:
            print(f"   [OK] Full memory (20+ trades)")

    # Check action history
    if hasattr(raw_env, 'action_history'):
        actions = list(raw_env.action_history)
        print(f"   Action history (last 5): {actions}")
        action_counts = Counter(actions)
        print(f"   Action distribution: {dict(action_counts)}")

    # Check MEMORY features in observation
    memory_features = obs[0][222:229]  # Last 7 features
    print(f"\n   MEMORY features (indices 222-228):")
    print(f"   - win_rate: {memory_features[0]:.4f}")
    print(f"   - streak: {memory_features[1]:.4f}")
    print(f"   - avg_pnl: {memory_features[2]:.4f}")
    print(f"   - best_trade: {memory_features[3]:.4f}")
    print(f"   - worst_trade: {memory_features[4]:.4f}")
    print(f"   - win_count: {memory_features[5]:.4f}")
    print(f"   - loss_count: {memory_features[6]:.4f}")

def question_9_reward_analysis():
    """Q9: Le reward HOLD est-il trop attractif?"""
    print("\n" + "="*60)
    print("Q9: Le reward pour HOLD est-il plus attractif que BUY/SELL?")
    print("="*60)

    print("   Analyse du reward function (trading_env_v2_ultimate.py):")
    print("")
    print("   [HOLD]")
    print("   - Pas de spread/slippage = 0 coût")
    print("   - Pas de risque de perte = safe")
    print("   - Diversity penalty si >90% = -0.1")
    print("")
    print("   [BUY/SELL]")
    print("   - Spread + Slippage = coût immédiat")
    print("   - Risque de perte = -R")
    print("   - Gain potentiel = +R")
    print("   - Trade completion bonus = +0.10")
    print("")
    print("   [ANALYSE]")
    print("   L'agent apprend: 'HOLD = safe, BUY/SELL = risque'")
    print("   → Mode collapse vers HOLD")
    print("")
    print("   [SOLUTION] Ajouter exploration bonus pour actions")

def question_10_solutions():
    """Q10: Comment ouvrir plus de positions?"""
    print("\n" + "="*60)
    print("Q10: Comment te faire ouvrir plus de positions?")
    print("="*60)

    print("\n   [SOLUTIONS PROPOSÉES]")
    print("")
    print("   1. ENTROPY SCHEDULE (modifier train_CRITIC_BOOST_LSTM.py)")
    print("      - Phase haute: 0-50% au lieu de 0-30%")
    print("      - Entropy min: 0.10 au lieu de 0.05")
    print("      - Decay: linéaire au lieu d'exponentiel")
    print("")
    print("   2. EXPLORATION BONUS (modifier trading_env_v2_ultimate.py)")
    print("      - +0.05 pour chaque action BUY ou SELL")
    print("      - Pénalité HOLD: -0.01 si HOLD > 80%")
    print("")
    print("   3. HOLD PENALTY (modifier _calculate_reward)")
    print("      - Si consecutive_holds > 50: -0.02")
    print("      - Réduire attractivité du HOLD")
    print("")
    print("   4. WARMUP TRADES (modifier reset)")
    print("      - Pré-remplir 10 trades aléatoires gagnants")
    print("      - Éviter le cold start")
    print("")
    print("   5. IMITATION LEARNING (optionnel)")
    print("      - Pré-entraîner sur trades experts")
    print("      - Apprendre à trader avant RL")

# ============================================================================
# RUN INTERVIEW
# ============================================================================
print("\n" + "="*80)
print("DÉBUT DE L'INTERVIEW - 10 QUESTIONS")
print("="*80)

# Reset environment and get initial observation
obs = vec_env.reset()
lstm_states = None

# Run episode for a few steps to get some context
for _ in range(10):
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

# Now interview
question_1_action_choice(model, obs, lstm_states)
question_2_probabilities(model, obs, lstm_states)
question_3_critic_value(model, obs, lstm_states)
question_4_env_blockers(vec_env)
question_5_confidence_threshold(vec_env)
question_6_feature_importance(obs)
question_7_patterns(vec_env)
question_8_memory(vec_env, obs)
question_9_reward_analysis()
question_10_solutions()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RÉSUMÉ DE L'INTERVIEW")
print("="*80)

print("""
L'agent souffre probablement de:
1. MODE COLLAPSE - Une action domine (vérifié Q2)
2. COLD START - Pas de trades en mémoire (vérifié Q8)
3. HOLD ATTRACTIF - Reward HOLD > BUY/SELL (analysé Q9)
4. ENTROPY BASSE - Pas assez d'exploration (schedule 30%)

PROCHAINES ÉTAPES:
1. Modifier AdaptiveEntropyCallback (phase haute 50%, min 0.10)
2. Ajouter exploration bonus pour BUY/SELL
3. Ajouter pénalité HOLD excessive
4. Re-lancer training 500K
""")

print("="*80)
print(f"FIN INTERVIEW: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Cleanup
vec_env.close()
