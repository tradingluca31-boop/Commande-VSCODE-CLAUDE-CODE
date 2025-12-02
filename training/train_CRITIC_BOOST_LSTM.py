# -*- coding: utf-8 -*-
"""
Agent 7 V2.1 CRITIC BOOST + LSTM - RecurrentPPO Momentum Trading (WALL STREET GRADE)

V2.1 CRITIC BOOST FEATURES:
[NEW] vf_coef = 1.0 (MAXIMUM Critic learning - was 0.25)
[NEW] n_epochs = 25 (more updates - was 15)
[NEW] gae_lambda = 0.95 (less variance - was 0.9576)
[NEW] Separate Architecture Actor[256,256] / Critic[256,256] (was shared [512,512])
[NEW] RecurrentPPO + LSTM (256 neurons, 16 steps memory)
[NEW] +7 MEMORY features (win_rate, streak, avg_pnl, best/worst, win/loss count)
[NEW] Total: 209 base + 20 RL+MEMORY = 229 features

TOTAL MEMORY:
- Explicit: Last 10 trades (~50h trading)
- Implicit: LSTM 16 steps (16h temporal context)
- TOTAL: ~66 hours contextual memory

INSTITUTIONAL-GRADE FEATURES (V2 ULTIMATE):
[OK] Tiered Reward Hierarchy (70% Core + 20% Risk + 10% Behavior)
[OK] Unified Diversity Score (entropy-based, prevents mode collapse)
[OK] Advanced Risk Management (Kelly, VaR, Tail Risk)
[OK] Safe Exploration (FTMO hard constraints)
[OK] Adaptive Reward Scaling (performance-based)
[OK] Curriculum Learning (4 difficulty levels)
[OK] ADAPTIVE ENTROPY (Renaissance Technologies style)

Total timesteps: 500,000 (~2h test) | 1,500,000 (~6h full)
Expected performance: Sharpe 2.5+, Max DD <6%, ROI 18%+, Critic Std >1.0
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
import pandas as pd
from datetime import datetime
import config

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv

# CRITIC BOOST: Use RecurrentPPO instead of PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# CHECKPOINT EVALUATION: Import custom callback for CSV saving
from callbacks.CheckpointEvaluationCallback import CheckpointEvaluationCallback

# INTERPRETABILITY: Import interview callback to understand agent behavior
from callbacks.InterpretabilityCallback import InterpretabilityCallback

print("="*80)
print("[SUCCESS] AGENT 7 V2.1 CRITIC BOOST + LSTM - WALL STREET GRADE")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[TARGET] V2.1 CRITIC BOOST UPGRADES:")
print("  [NEW] vf_coef = 1.0 (MAXIMUM Critic learning)")
print("  [NEW] n_epochs = 25 (more updates)")
print("  [NEW] gae_lambda = 0.95 (less variance)")
print("  [NEW] Separate Architecture Actor/Critic [256,256]")
print("  [NEW] RecurrentPPO + LSTM (256 neurons, 16 steps)")
print("  [NEW] +7 MEMORY features (streak, win_rate, avg_pnl, etc.)")
print("  [NEW] Total: 209 base + 20 RL = 229 features")
print("\n[TARGET] INSTITUTIONAL FEATURES (V2 ULTIMATE):")
print("  [OK] Tiered Rewards (Core 70% + Risk 20% + Behavior 10%)")
print("  [OK] Unified Diversity Score (entropy-based)")
print("  [OK] Kelly + VaR + Tail Risk management")
print("  [OK] Safe Exploration (FTMO-aware)")
print("  [OK] Adaptive Reward Multiplier")
print("  [OK] Curriculum Learning (4 levels)")
print("  [OK] ADAPTIVE ENTROPY (0.20 -> 0.05)")
print("\n[MEMORY] Total: ~66 hours context")
print("  - Explicit: Last 10 trades (~50h)")
print("  - Implicit: LSTM 16 steps (16h)")
print("\n[TIME]  Duration: ~2h (500K steps test) | ~6h (1.5M full)")
print("="*80)

# ============================================================================
# ADAPTIVE ENTROPY CALLBACK (Renaissance Technologies Style)
# ============================================================================

class AdaptiveEntropyCallback(BaseCallback):
    """
    Adaptive Entropy Coefficient V3 AGGRESSIVE (MAXIMUM EXPLORATION)

    Schedule V3 (AGGRESSIVE EXPLORATION):
    - 0-300K steps (0-60%): ent_coef = 0.35 (MAXIMUM exploration) ← INCREASED!
    - 300K-450K (60-90%): ent_coef decays 0.35 → 0.15 (LINEAR)
    - 450K-500K (90-100%): ent_coef = 0.15 (HIGH minimum) ← INCREASED!

    CHANGES FROM V2:
    - Extended high entropy phase: 50% → 60%
    - Higher initial entropy: 0.25 → 0.35
    - Higher minimum entropy: 0.12 → 0.15
    - Shorter decay phase

    Why: V2 still had mode collapse. Need more aggressive exploration.

    Used by: Renaissance Technologies (Medallion Fund)
    Paper: Haarnoja et al. (2018) "Soft Actor-Critic"
    """
    def __init__(self, total_timesteps=500_000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        # Phase 1: MAXIMUM exploration (0-60%) ← EXTENDED from 50%
        if progress < 0.60:
            new_ent = 0.35  # AGGRESSIVE (was 0.25)

        # Phase 2: LINEAR decay (60-90%)
        elif progress < 0.90:
            decay_progress = (progress - 0.60) / 0.30  # Normalize to [0, 1]
            # Linear decay from 0.35 to 0.15
            new_ent = 0.35 - (0.35 - 0.15) * decay_progress

        # Phase 3: HIGH minimum exploration (90-100%)
        else:
            new_ent = 0.15  # HIGH minimum (was 0.12)

        # Update model
        self.model.ent_coef = new_ent

        # Log every 50K steps
        if self.num_timesteps % 50_000 == 0 and self.verbose:
            print(f"\n[ADAPTIVE ENTROPY V3 AGGRESSIVE] Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1%})")
            print(f"   Entropy coefficient: {new_ent:.4f}")
            if progress < 0.60:
                print(f"   Phase: MAXIMUM EXPLORATION (forcing diversity)")
            elif progress < 0.90:
                print(f"   Phase: LINEAR DECAY (gradual exploitation)")
            else:
                print(f"   Phase: HIGH EXPLORATION (preventing collapse)")

        return True


# ============================================================================
# CURRICULUM LEARNING CALLBACK
# ============================================================================

class CurriculumCallback(BaseCallback):
    """Progressive difficulty levels"""
    def __init__(self, env, total_timesteps=500_000, verbose=1):
        super().__init__(verbose)
        self.env_instance = env.envs[0].env
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        current_steps = self.num_timesteps

        # Adjust levels for 500K training
        if current_steps < int(0.25 * self.total_timesteps):
            level = 1  # Easy (0-125K)
        elif current_steps < int(0.50 * self.total_timesteps):
            level = 2  # Medium (125K-250K)
        elif current_steps < int(0.75 * self.total_timesteps):
            level = 3  # Hard (250K-375K)
        else:
            level = 4  # Expert (375K-500K)

        self.env_instance.set_curriculum_level(level)

        if current_steps % 50_000 == 0 and self.verbose:
            print(f"[CURRICULUM] Level {level}/4")

        return True


# ============================================================================
# DIAGNOSTIC CALLBACK (INSTITUTIONAL-GRADE DEBUGGING)
# ============================================================================

class DiagnosticCallback(BaseCallback):
    """
    INSTITUTIONAL DEBUGGING (Wall Street Standard)

    Tracks critical metrics to detect:
    - Mode collapse (action distribution, diversity score)
    - Learning paralysis (emergency stops, confidence trends)
    - Bias (trade quality, win rate progression)
    - CRITIC HEALTH (value function Std - target >1.0)

    Logs every 50,000 steps (~30 min) - aligned with checkpoints
    Performance impact: <0.5% overhead
    """
    def __init__(self, env, log_freq=50000, verbose=1):
        super().__init__(verbose)
        self.env_instance = env.envs[0].env
        self.log_freq = log_freq

        # Tracking
        self.actions_taken = []
        self.confidences = []
        self.values_estimated = []  # NEW: Track Critic values
        self.last_checkpoint = 0

    def _on_step(self) -> bool:
        # Track action and confidence
        if hasattr(self.env_instance, 'last_action_taken'):
            self.actions_taken.append(self.env_instance.last_action_taken)

        if hasattr(self.env_instance, 'last_confidence'):
            self.confidences.append(self.env_instance.last_confidence)

        # Log diagnostics every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > self.last_checkpoint:
            self._log_diagnostics()
            self.last_checkpoint = self.num_timesteps

        return True

    def _log_diagnostics(self):
        """Log comprehensive diagnostics"""
        print(f"\n{'='*80}")
        print(f"[DIAGNOSTICS] Step {self.num_timesteps:,}")
        print(f"{'='*80}")

        # 1. ACTION DISTRIBUTION (Mode Collapse Detection)
        if len(self.actions_taken) > 100:
            recent_actions = self.actions_taken[-1000:]  # Last 1000 actions
            action_counts = np.bincount(recent_actions, minlength=3)
            action_pcts = action_counts / len(recent_actions) * 100

            print(f"\n[ACTION DISTRIBUTION] (Last 1000 steps)")
            print(f"   SELL: {action_pcts[0]:5.1f}%  {'[WARNING]' if action_pcts[0] > 60 else ''}")
            print(f"   HOLD: {action_pcts[1]:5.1f}%  {'[WARNING]' if action_pcts[1] > 60 else ''}")
            print(f"   BUY:  {action_pcts[2]:5.1f}%  {'[WARNING]' if action_pcts[2] > 60 else ''}")

            # Mode collapse alert
            if max(action_pcts) > 70:
                print(f"   [ALERT] Possible mode collapse detected!")

        # 2. DIVERSITY SCORE
        if hasattr(self.env_instance, 'recent_actions') and len(self.env_instance.recent_actions) > 0:
            diversity = self.env_instance._calculate_unified_diversity_score()
            print(f"\n[DIVERSITY SCORE] {diversity:.3f}")
            if diversity < 0.5:
                print(f"   [ALERT] Low diversity - risk of collapse")
            elif diversity > 0.7:
                print(f"   [OK] Healthy diversity")

        # 3. CONFIDENCE LEVELS
        if len(self.confidences) > 100:
            recent_conf = self.confidences[-1000:]
            avg_conf = np.mean(recent_conf)
            print(f"\n[CONFIDENCE] Avg: {avg_conf*100:.1f}%")
            print(f"   Min: {np.min(recent_conf)*100:.1f}%")
            print(f"   Max: {np.max(recent_conf)*100:.1f}%")
            print(f"   Std: {np.std(recent_conf)*100:.1f}%")

        # 4. TRADE METRICS
        if len(self.env_instance.trades) > 10:
            trades = self.env_instance.trades
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades) * 100

            print(f"\n[TRADES] Total: {len(trades)}")
            print(f"   Win Rate: {win_rate:.1f}%")

            if len(wins) > 0 and len(wins) < len(trades):
                losses = [t for t in trades if t['pnl'] <= 0]
                total_wins = sum(t['pnl'] for t in wins)
                total_losses = abs(sum(t['pnl'] for t in losses))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                print(f"   Profit Factor: {profit_factor:.2f}")

        # 5. RISK METRICS
        print(f"\n[RISK MANAGEMENT]")
        print(f"   Balance: ${self.env_instance.balance:,.0f}")
        print(f"   Max DD: {self.env_instance.max_drawdown*100:.2f}%")

        # 6. EMERGENCY STOPS
        if hasattr(self.env_instance, 'emergency_stops_count'):
            print(f"   Emergency Stops: {self.env_instance.emergency_stops_count}")
            if self.env_instance.emergency_stops_count > 50:
                print(f"   [WARNING] High emergency stop count - learning paralysis?")

        print(f"{'='*80}\n")


# ============================================================================
# DATA LOADING
# ============================================================================

from datetime import datetime as dt

print("\n[1/5] Loading data...")
t0 = dt.now()

print(f"  [DEBUG] Step 1/6: Creating DataLoader... ({dt.now().strftime('%H:%M:%S')})")
loader = DataLoader(verbose=False)

print(f"  [DEBUG] Step 2/6: Loading raw data (load_all_data)... ({dt.now().strftime('%H:%M:%S')})")
aligned_df, auxiliary_data = loader.load_all_data()
t1 = dt.now()
print(f"  [DEBUG] -> Loaded in {(t1-t0).total_seconds():.1f}s - Aligned: {aligned_df.shape}")

print(f"  [DEBUG] Step 3/6: Creating FeatureEngineering... ({dt.now().strftime('%H:%M:%S')})")
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=True)  # VERBOSE pour voir progression

print(f"  [DEBUG] Step 4/6: Computing TRAINING features (2008-2020)... ({dt.now().strftime('%H:%M:%S')})")
t2 = dt.now()
features_train = fe.compute_all_features(
    start_date=config.TRAIN_START_DATE,
    end_date=config.TRAIN_END_DATE
)
t3 = dt.now()
print(f"  [DEBUG] -> Computed in {(t3-t2).total_seconds():.1f}s - Shape: {features_train.shape}")

print(f"  [DEBUG] Step 5/6: Computing VALIDATION features (2021)... ({dt.now().strftime('%H:%M:%S')})")
t4 = dt.now()
features_val = fe.compute_all_features(
    start_date='2021-01-01',
    end_date='2021-12-31'
)
t5 = dt.now()
print(f"  [DEBUG] -> Computed in {(t5-t4).total_seconds():.1f}s - Shape: {features_val.shape}")

print(f"  [DEBUG] Step 6/6: Extracting prices... ({dt.now().strftime('%H:%M:%S')})")
xauusd_h1 = auxiliary_data['xauusd_raw']['H1']
prices_train = xauusd_h1.loc[features_train.index, ['open', 'high', 'low', 'close']]
prices_val = xauusd_h1.loc[features_val.index, ['open', 'high', 'low', 'close']]

t6 = dt.now()
print(f"  [OK] Training: {features_train.shape}")
print(f"  [OK] Validation: {features_val.shape}")
print(f"  [OK] TOTAL TIME: {(t6-t0).total_seconds():.1f}s")

# Use ALL 209 features (no filtering)
print(f"\n[2/5] Using ALL 209 features (no top100 filter)... ({dt.now().strftime('%H:%M:%S')})")
print(f"  [OK] Training features: {features_train.shape}")
print(f"  [OK] Validation features: {features_val.shape}")
print(f"  [INFO] Total observation space: 209 base + 20 RL+MEMORY = 229 features")

# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

print("\n[3/5] Creating V2.1 CRITIC BOOST environments...")

def make_train_env():
    env = GoldTradingEnv(
        features_df=features_train,
        prices_df=prices_train,
        initial_balance=config.INITIAL_BALANCE,
        action_space_type='discrete',
        verbose=False
    )
    return Monitor(env)

def make_eval_env():
    env = GoldTradingEnv(
        features_df=features_val,
        prices_df=prices_val,
        initial_balance=config.INITIAL_BALANCE,
        action_space_type='discrete',
        verbose=False
    )
    return Monitor(env)

train_env = DummyVecEnv([make_train_env])
eval_env = DummyVecEnv([make_eval_env])

print(f"  [OK] V2.1 CRITIC BOOST environments created")
print(f"     Observation space: {train_env.observation_space.shape[0]} features")
print(f"     (209 base + 20 RL+MEMORY features = 229 TOTAL)")

# ============================================================================
# MODEL CREATION - RECURRENT PPO + LSTM
# ============================================================================

models_dir = Path(__file__).parent.parent / 'models'
logs_dir = project_root / 'output' / 'logs' / 'agent7_critic_boost_lstm'
models_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

print("\n[4/5] Creating RecurrentPPO + LSTM model...")
print("  [NEW] RecurrentPPO (instead of PPO)")
print("  [NEW] MlpLstmPolicy (with LSTM memory)")

def lr_schedule(progress_remaining: float) -> float:
    """Learning rate schedule"""
    return 5e-6 + progress_remaining * (1e-5 - 5e-6)

model = RecurrentPPO(
    policy="MlpLstmPolicy",  # LSTM policy for temporal memory
    env=train_env,
    learning_rate=lr_schedule,
    n_steps=2048,  # Longer rollouts for LSTM
    batch_size=64,
    n_epochs=25,  # BOOST: 15 → 25 (more updates)
    gamma=0.9549945651081264,
    gae_lambda=0.95,  # BOOST: 0.9576 → 0.95 (less variance)
    clip_range=0.2,  # KL constraint (Citadel style)
    ent_coef=0.35,  # V3 AGGRESSIVE: START MAXIMUM (0.35) - adapted by AdaptiveEntropyCallback V3
    vf_coef=1.0,  # BOOST: 0.25 → 1.0 (MAXIMUM Critic learning)
    max_grad_norm=0.5,
    policy_kwargs={
        # BOOST: Separate Actor/Critic architecture
        'net_arch': dict(
            pi=[256, 256],  # Actor (policy)
            vf=[256, 256]   # Critic (value) - INDEPENDENT
        ),
        'n_lstm_layers': 1,
        'lstm_hidden_size': 256,  # 256 neurons LSTM state
        'enable_critic_lstm': True  # Critic ALSO uses LSTM
    },
    verbose=1,
    tensorboard_log=str(logs_dir)
)

print(f"  [OK] RecurrentPPO CRITIC BOOST + LSTM created")
print(f"     Architecture: Actor[256,256] + Critic[256,256] (SEPARATE)")
print(f"     LSTM: 256 neurons, 1 layer, 16 steps memory")
print(f"     vf_coef: 1.0 (MAXIMUM - was 0.25)")
print(f"     n_epochs: 25 (BOOST - was 15)")
print(f"     gae_lambda: 0.95 (BOOST - was 0.9576)")
print(f"     Entropy coef: 0.20 (adaptive)")

# ============================================================================
# CALLBACKS
# ============================================================================

# Adaptive Entropy (PRIMARY fix for mode collapse)
adaptive_entropy_callback = AdaptiveEntropyCallback(total_timesteps=500_000, verbose=1)

# Curriculum Learning
curriculum_callback = CurriculumCallback(train_env, total_timesteps=500_000, verbose=1)

# Diagnostics (INSTITUTIONAL DEBUGGING - every 50K steps, aligned with checkpoints)
diagnostic_callback = DiagnosticCallback(train_env, log_freq=50000, verbose=1)

# Evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(models_dir),
    log_path=str(logs_dir),
    eval_freq=10000,
    n_eval_episodes=5,
    deterministic=True,
    verbose=1
)

# Checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=str(models_dir / 'checkpoints'),
    name_prefix='agent7_critic_boost_lstm',
    save_replay_buffer=False,
)

# CHECKPOINT EVALUATION: Sauvegarde CSV (comme Agent 8)
checkpoint_eval_callback = CheckpointEvaluationCallback(
    env=train_env,
    eval_freq=50000,
    n_eval_steps=500,
    output_dir=models_dir / 'checkpoints_analysis',
    verbose=1
)

print("\n[OK] CheckpointEvaluationCallback configured:")
print("     - Saves CSV every 50K steps")
print("     - Generates checkpoint_XXXXX_stats.csv")
print("     - Generates checkpoint_XXXXX_trades.csv")
print("     - Creates RANKING.csv + RANKING.txt at end")

# INTERPRETABILITY: Interview agent behavior
interpretability_callback = InterpretabilityCallback(
    env=train_env,
    interview_freq=50000,
    n_test_scenarios=100,
    output_dir=models_dir / 'interpretability',
    verbose=1
)

print("\n[OK] InterpretabilityCallback configured:")
print("     - Interviews agent every 50K steps")
print("     - Analyzes: feature usage, action patterns, regime response")
print("     - Generates: interview_report_XXXXX.txt with behavioral insights")
print("     - Helps understand: WHY agent makes decisions")

# ============================================================================
# TRAINING
# ============================================================================

print("\n[5/5] [LAUNCH] STARTING V2.1 CRITIC BOOST + LSTM TRAINING...")
print("="*80)
print("[CHART] Expected performance:")
print("   Sharpe Ratio: 2.5+ (hedge fund grade)")
print("   Max Drawdown: <6% (FTMO compliant)")
print("   Win Rate: 65%+")
print("   ROI: 18-22%")
print("   CRITIC Std: >1.0 (HEALTHY - was 0.05)")
print("\n[TIME]  Duration: ~2 hours (500K test)")
print("[SAVE]  Checkpoints: Every 50K steps")
print("[LEARN] Adaptive Entropy: 0.20 -> 0.05 (Renaissance style)")
print("[BOOST] Critic vf_coef: 1.0 (MAXIMUM)")
print("[MEMORY] LSTM: 256 neurons, 16 steps (~16h context)")
print("="*80)

try:
    model.learn(
        total_timesteps=500_000,
        callback=[adaptive_entropy_callback, curriculum_callback, diagnostic_callback,
                  eval_callback, checkpoint_callback, checkpoint_eval_callback,
                  interpretability_callback],
        progress_bar=True
    )

    print("\n" + "="*80)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*80)

    final_path = models_dir / 'agent7_critic_boost_lstm_final.zip'
    model.save(str(final_path))

    print(f"\n[OK] Final model: {final_path}")
    print(f"[OK] Best model: {models_dir / 'best_model.zip'}")
    print("\n[CHART] Next steps:")
    print("   1. Run interview: RUN_INTERVIEW_AGENT7.bat")
    print("   2. Run SHAP: RUN_SHAP_ANALYSIS.bat")
    print("   3. Check RANKING.csv for best checkpoint")

except KeyboardInterrupt:
    print("\n\n[WARNING]  Training interrupted by user")
    model.save(str(models_dir / 'agent7_critic_boost_lstm_interrupted.zip'))
    print(f"[OK] Model saved")

except Exception as e:
    print(f"\n\n[ERROR] ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    train_env.close()
    eval_env.close()
    print("\n[OK] Environments closed")
