# -*- coding: utf-8 -*-
"""
================================================================================
INTERVIEW INTERACTIF - AGENT 7 V2.1
================================================================================

Pose des questions directement à l'agent et obtiens ses réponses!

L'agent "répond" en montrant:
- Ses probabilités d'action (ce qu'il PENSE)
- Son action choisie (ce qu'il FAIT)
- Sa Value Function (comment il ÉVALUE la situation)
- Les features qui l'influencent (POURQUOI il décide)

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

# ============================================================================
# SETUP
# ============================================================================

def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("\n" + "="*70)
    print("   🤖 INTERVIEW AGENT 7 V2.1 - CRITIC BOOST + LSTM")
    print("="*70)

def get_unwrapped_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env

# ============================================================================
# LOAD EVERYTHING
# ============================================================================

print("Chargement de l'agent en cours...")
print("(Cela peut prendre 30 secondes)")

# Load data
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=False)
features = fe.compute_all_features(config.TEST_START_DATE, config.TEST_END_DATE)
prices = auxiliary_data['xauusd_raw']['H1'].loc[features.index, ['open', 'high', 'low', 'close']]

# Create environment
env = GoldTradingEnv(
    features_df=features,
    prices_df=prices,
    initial_balance=100_000,
    action_space_type='discrete',
    verbose=False
)
env_monitor = Monitor(env)
vec_env = DummyVecEnv([lambda: env_monitor])

# Load model
model_paths = [
    parent_dir / "models" / "checkpoints" / "agent7_critic_boost_lstm_150000_steps.zip",
    parent_dir / "models" / "best_model.zip",
    parent_dir / "models" / "agent7_critic_boost_lstm_final.zip"
]

model = None
model_name = "Unknown"
for path in model_paths:
    if path.exists():
        model = RecurrentPPO.load(path, env=vec_env)
        model_name = path.name
        break

if model is None:
    print("ERREUR: Aucun modèle trouvé!")
    sys.exit(1)

# Initialize
obs = vec_env.reset()
lstm_states = None
raw_env = get_unwrapped_env(vec_env)

print(f"\n✅ Agent chargé: {model_name}")
print("✅ Environnement prêt")

# ============================================================================
# INTERVIEW FUNCTIONS
# ============================================================================

def ask_action():
    """Demande: Quelle action vas-tu prendre?"""
    global obs, lstm_states

    print("\n" + "-"*50)
    print("❓ QUESTION: Quelle action vas-tu prendre maintenant?")
    print("-"*50)

    # Get probabilities by sampling
    action_counts = Counter()
    for _ in range(100):
        action, _ = model.predict(obs, state=lstm_states, deterministic=False)
        action_counts[int(action[0])] += 1

    probs = {k: v/100 for k, v in action_counts.items()}

    # Get deterministic action
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    action_int = int(action[0])
    action_names = {0: "SELL 📉", 1: "HOLD ⏸️", 2: "BUY 📈"}

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")
    print(f"   Ma décision: {action_names[action_int]}")
    print(f"")
    print(f"   Mes probabilités:")
    print(f"   ├── SELL: {'█' * int(probs.get(0,0)*20):<20} {probs.get(0,0)*100:.1f}%")
    print(f"   ├── HOLD: {'█' * int(probs.get(1,0)*20):<20} {probs.get(1,0)*100:.1f}%")
    print(f"   └── BUY:  {'█' * int(probs.get(2,0)*20):<20} {probs.get(2,0)*100:.1f}%")

    # Interpretation
    max_prob = max(probs.values()) if probs else 0
    if max_prob > 0.9:
        print(f"\n   ⚠️ Je suis TRÈS sûr de moi ({max_prob*100:.0f}%)")
        print(f"      Risque de mode collapse!")
    elif max_prob > 0.6:
        print(f"\n   💭 J'ai une préférence claire ({max_prob*100:.0f}%)")
    else:
        print(f"\n   🤔 J'hésite entre plusieurs options")

    return action_int

def ask_why():
    """Demande: Pourquoi cette décision?"""
    global obs

    print("\n" + "-"*50)
    print("❓ QUESTION: Pourquoi cette décision?")
    print("-"*50)

    obs_flat = obs.flatten()

    # Get top influential features
    top_indices = np.argsort(np.abs(obs_flat))[::-1][:10]

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")
    print(f"   Les features qui m'influencent le plus:")
    print(f"")

    for i, idx in enumerate(top_indices[:5]):
        val = obs_flat[idx]
        direction = "↑ haut" if val > 0 else "↓ bas"
        print(f"   {i+1}. Feature[{idx}]: {val:+.3f} ({direction})")

    # Check RL features
    print(f"\n   Mon état interne (RL features):")
    rl_start = 209
    print(f"   ├── Position actuelle: {obs_flat[220]:.0f} (-1=SHORT, 0=FLAT, 1=LONG)")
    print(f"   ├── Regret signal: {obs_flat[212]:.3f}")
    print(f"   ├── Volatility: {obs_flat[219]:.3f}")
    print(f"   └── Win rate récent: {obs_flat[222]:.3f}")

def ask_fear():
    """Demande: As-tu peur de perdre?"""
    global obs, lstm_states

    print("\n" + "-"*50)
    print("❓ QUESTION: As-tu peur de perdre de l'argent?")
    print("-"*50)

    # Analyze HOLD preference
    action_counts = Counter()
    for _ in range(100):
        action, _ = model.predict(obs, state=lstm_states, deterministic=False)
        action_counts[int(action[0])] += 1

    hold_pct = action_counts.get(1, 0) / 100

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")

    if hold_pct > 0.8:
        print(f"   😰 OUI, j'ai très peur!")
        print(f"   Je choisis HOLD {hold_pct*100:.0f}% du temps")
        print(f"   HOLD = pas de risque = pas de perte")
        print(f"")
        print(f"   💭 Mon raisonnement:")
        print(f"   'Si je ne trade pas, je ne peux pas perdre'")
        print(f"   'Le marché est dangereux, mieux vaut attendre'")
    elif hold_pct > 0.5:
        print(f"   😐 Un peu, je suis prudent")
        print(f"   Je choisis HOLD {hold_pct*100:.0f}% du temps")
        print(f"   Je préfère attendre les bonnes opportunités")
    else:
        print(f"   😎 Non, je suis confiant!")
        print(f"   Je ne choisis HOLD que {hold_pct*100:.0f}% du temps")
        print(f"   Je n'ai pas peur de prendre des risques calculés")

def ask_position():
    """Demande: Quelle est ta position actuelle?"""
    global raw_env

    print("\n" + "-"*50)
    print("❓ QUESTION: Quelle est ta position actuelle?")
    print("-"*50)

    pos = raw_env.position_side
    pos_names = {-1: "SHORT 📉", 0: "FLAT (pas de position)", 1: "LONG 📈"}

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")
    print(f"   Position: {pos_names[pos]}")
    print(f"   Balance: ${raw_env.balance:,.2f}")
    print(f"   Equity: ${raw_env.equity:,.2f}")
    print(f"   Drawdown: {raw_env.max_drawdown*100:.2f}%")
    print(f"   Trades effectués: {len(raw_env.trades)}")

    if pos == 0:
        print(f"\n   💭 Je n'ai pas de position ouverte")
        print(f"      J'attends une opportunité...")
    else:
        print(f"\n   💭 Je suis en position {pos_names[pos]}")
        print(f"      Entry price: ${raw_env.entry_price:.2f}")

def ask_memory():
    """Demande: Que retiens-tu de tes trades passés?"""
    global raw_env, obs

    print("\n" + "-"*50)
    print("❓ QUESTION: Que retiens-tu de tes trades passés?")
    print("-"*50)

    n_trades = len(raw_env.trades)
    obs_flat = obs.flatten()

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")
    print(f"   Nombre de trades en mémoire: {n_trades}")

    if n_trades == 0:
        print(f"\n   😕 Je n'ai aucun souvenir de trades!")
        print(f"      C'est un COLD START")
        print(f"      Mes MEMORY features sont à valeur par défaut")
        print(f"")
        print(f"   Mes features mémoire actuelles:")
        print(f"   ├── Win rate: {obs_flat[222]:.2f} (défaut: 0.5)")
        print(f"   ├── Streak: {obs_flat[223]:.2f} (défaut: 0)")
        print(f"   ├── Avg PnL: {obs_flat[224]:.2f} (défaut: 0)")
        print(f"   └── Best/Worst: {obs_flat[225]:.2f}/{obs_flat[226]:.2f}")
    else:
        wins = sum(1 for t in raw_env.trades if t['pnl'] > 0)
        losses = n_trades - wins

        print(f"\n   📊 Mon historique:")
        print(f"   ├── Wins: {wins}")
        print(f"   ├── Losses: {losses}")
        print(f"   └── Win Rate: {wins/n_trades*100:.1f}%")

        if n_trades >= 3:
            recent = raw_env.trades[-3:]
            print(f"\n   📈 Mes 3 derniers trades:")
            for i, t in enumerate(recent):
                result = "✅ WIN" if t['pnl'] > 0 else "❌ LOSS"
                print(f"   {i+1}. {result}: ${t['pnl']:.2f}")

def ask_execute_step():
    """Exécute un step et montre ce qui se passe"""
    global obs, lstm_states, raw_env

    print("\n" + "-"*50)
    print("❓ QUESTION: Montre-moi un step complet")
    print("-"*50)

    # Before
    pos_before = raw_env.position_side
    balance_before = raw_env.balance

    # Get action
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    action_int = int(action[0])
    action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

    print(f"\n🤖 EXÉCUTION DU STEP:")
    print(f"")
    print(f"   1️⃣ Je reçois l'observation (229 features)")
    print(f"   2️⃣ Mon LSTM traite la séquence")
    print(f"   3️⃣ Je décide: {action_names[action_int]}")

    # Execute
    obs, reward, done, info = vec_env.step(action)

    # After
    pos_after = raw_env.position_side
    balance_after = raw_env.balance

    print(f"   4️⃣ L'environnement exécute mon action")
    print(f"   5️⃣ Je reçois reward: {reward[0]:+.4f}")
    print(f"")
    print(f"   📊 Résultat:")
    print(f"   ├── Position: {pos_before} → {pos_after}")
    print(f"   ├── Balance: ${balance_before:,.2f} → ${balance_after:,.2f}")
    print(f"   └── Changement: ${balance_after - balance_before:+.2f}")

    if pos_before == 0 and pos_after != 0:
        print(f"\n   🎯 J'ai OUVERT une position!")
    elif pos_before != 0 and pos_after == 0:
        print(f"\n   🏁 J'ai FERMÉ ma position!")
    elif action_int == 1:
        print(f"\n   ⏸️ J'ai choisi de ne rien faire (HOLD)")

def ask_blockers():
    """Demande: Qu'est-ce qui t'empêche de trader?"""
    global raw_env

    print("\n" + "-"*50)
    print("❓ QUESTION: Qu'est-ce qui t'empêche de trader?")
    print("-"*50)

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")
    print(f"   Vérification des blocages...")
    print(f"")

    blockers = []

    # Daily loss
    if hasattr(raw_env, 'daily_loss_limit_reached') and raw_env.daily_loss_limit_reached:
        blockers.append("🚫 Daily Loss Limit atteint!")
    else:
        print(f"   ✅ Daily Loss Limit: OK")

    # Max drawdown
    if hasattr(raw_env, 'max_drawdown'):
        dd = raw_env.max_drawdown
        if dd >= 0.10:
            blockers.append(f"🚫 Max Drawdown >= 10% ({dd*100:.1f}%)")
        elif dd >= 0.07:
            print(f"   ⚠️ Drawdown élevé: {dd*100:.1f}% (attention!)")
        else:
            print(f"   ✅ Drawdown: {dd*100:.1f}% (OK)")

    # Tail risk
    if hasattr(raw_env, 'tail_risk_detected') and raw_env.tail_risk_detected:
        blockers.append("🚫 Tail Risk détecté!")
    else:
        print(f"   ✅ Tail Risk: Non détecté")

    # Confidence threshold
    if hasattr(raw_env, '_calculate_min_confidence_threshold'):
        threshold = raw_env._calculate_min_confidence_threshold()
        print(f"   ℹ️ Seuil de confiance: {threshold*100:.0f}%")
        print(f"      (Pour discrete, size=100% donc pas de blocage)")

    if blockers:
        print(f"\n   🚨 BLOCAGES ACTIFS:")
        for b in blockers:
            print(f"   {b}")
    else:
        print(f"\n   ✅ Aucun blocage technique!")
        print(f"   Si je ne trade pas, c'est par CHOIX")

def ask_improve():
    """Demande: Comment puis-je t'aider à trader plus?"""
    print("\n" + "-"*50)
    print("❓ QUESTION: Comment puis-je t'aider à trader plus?")
    print("-"*50)

    print(f"\n🤖 RÉPONSE DE L'AGENT:")
    print(f"")
    print(f"   💡 Pour me faire trader plus, il faudrait:")
    print(f"")
    print(f"   1. 🎲 PLUS D'EXPLORATION")
    print(f"      Augmenter mon entropy coefficient")
    print(f"      Actuellement: 0.25 (déjà augmenté!)")
    print(f"")
    print(f"   2. 🎁 RÉCOMPENSER MES TRADES")
    print(f"      Ajouter un bonus quand je BUY ou SELL")
    print(f"      Actuellement: +0.03 par action (déjà ajouté!)")
    print(f"")
    print(f"   3. 😤 PÉNALISER MON INACTION")
    print(f"      Pénaliser si je fais trop de HOLD")
    print(f"      Actuellement: -0.02 si >30 HOLDs (déjà ajouté!)")
    print(f"")
    print(f"   4. 🧠 ME DONNER DE L'EXPÉRIENCE")
    print(f"      Pré-remplir ma mémoire avec des trades")
    print(f"      Pour éviter le cold start")
    print(f"")
    print(f"   📋 FIXES DÉJÀ IMPLÉMENTÉS:")
    print(f"   ✅ Entropy étendue: 0-50% à 0.25")
    print(f"   ✅ Entropy minimum: 0.12 (au lieu de 0.05)")
    print(f"   ✅ Exploration bonus: +0.03")
    print(f"   ✅ HOLD penalty: -0.02")

# ============================================================================
# MAIN MENU
# ============================================================================

def main_menu():
    print_header()
    print(f"\n   Modèle: {model_name}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n" + "="*70)
    print("\n   QUESTIONS DISPONIBLES:")
    print(f"")
    print(f"   [1] 🎯 Quelle action vas-tu prendre?")
    print(f"   [2] 🤔 Pourquoi cette décision?")
    print(f"   [3] 😰 As-tu peur de perdre?")
    print(f"   [4] 📊 Quelle est ta position actuelle?")
    print(f"   [5] 🧠 Que retiens-tu de tes trades?")
    print(f"   [6] ▶️  Montre-moi un step complet")
    print(f"   [7] 🚫 Qu'est-ce qui t'empêche de trader?")
    print(f"   [8] 💡 Comment t'aider à trader plus?")
    print(f"")
    print(f"   [A] 🔄 Toutes les questions (automatique)")
    print(f"   [R] 🔁 Reset l'environnement")
    print(f"   [Q] 🚪 Quitter")
    print(f"\n" + "="*70)

def run_all_questions():
    """Exécute toutes les questions automatiquement"""
    print("\n" + "="*70)
    print("   INTERVIEW COMPLÈTE - TOUTES LES QUESTIONS")
    print("="*70)

    ask_action()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_why()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_fear()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_position()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_memory()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_execute_step()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_blockers()
    input("\n   [Appuyez sur Entrée pour continuer...]")

    ask_improve()

# ============================================================================
# MAIN LOOP
# ============================================================================

if __name__ == "__main__":
    while True:
        main_menu()

        choice = input("\n   Votre choix: ").strip().upper()

        if choice == '1':
            ask_action()
        elif choice == '2':
            ask_why()
        elif choice == '3':
            ask_fear()
        elif choice == '4':
            ask_position()
        elif choice == '5':
            ask_memory()
        elif choice == '6':
            ask_execute_step()
        elif choice == '7':
            ask_blockers()
        elif choice == '8':
            ask_improve()
        elif choice == 'A':
            run_all_questions()
        elif choice == 'R':
            obs = vec_env.reset()
            lstm_states = None
            print("\n   ✅ Environnement réinitialisé!")
        elif choice == 'Q':
            print("\n   👋 Au revoir!")
            break
        else:
            print("\n   ❌ Choix invalide, réessayez.")

        input("\n   [Appuyez sur Entrée pour continuer...]")

# Cleanup
vec_env.close()
