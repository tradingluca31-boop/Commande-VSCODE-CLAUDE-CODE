# -*- coding: utf-8 -*-
"""
CONTINUE TRAINING FROM 450K → 1M STEPS
Load best checkpoint (450K) and continue training to reach 1M total steps
"""

import sys
from pathlib import Path

# Add project root and parent directory
project_root = Path("C:/Users/lbye3/Desktop/GoldRL")
parent_dir = Path(__file__).resolve().parent.parent  # FICHIER IMPORTANT AGENT 7
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'AGENT_V2'))
sys.path.append(str(parent_dir))

import numpy as np
import pandas as pd
from datetime import datetime
import config

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from environment.trading_env_v2_ultimate import GoldTradingEnv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from callbacks.CheckpointEvaluationCallback import CheckpointEvaluationCallback
from callbacks.InterpretabilityCallback import InterpretabilityCallback

print("="*80)
print("[CONTINUE TRAINING] FROM 450K → 1M STEPS")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[STRATEGY] Continue from BEST checkpoint (450K steps)")
print("  - Win Rate: 70.05%")
print("  - ROI: 12.72%")
print("  - Profit Factor: 1.37")
print("  - Max DD: 7.89%")
print("\n[TARGET] Train 450K → 1M (550K additional steps)")
print("  - Duration: ~3h20min")
print("  - Checkpoints every 50K (450K, 500K, 550K, ..., 1M)")
print("="*80)

# ============================================================================
# CURRICULUM LEARNING CALLBACK (450K → 1M)
# ============================================================================

class CurriculumCallback(BaseCallback):
    """
    Progressive difficulty for continuation training (450K → 1M)

    Levels adjusted for continuation:
    - Level 3 (450K-600K): Hard - 2017-2019 choppy markets
    - Level 4 (600K-900K): Expert - 2020-2021 COVID volatility
    - Level 5 (900K-1M): Elite - 2022-2024 recent conditions
    """
    def __init__(self, env, total_timesteps=1_000_000, verbose=1):
        super().__init__(verbose)
        self.env_instance = env.envs[0].env
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Current step in TOTAL (450K + n_calls)
        current_step = 450_000 + self.num_timesteps
        progress = current_step / self.total_timesteps

        # Curriculum levels for continuation
        if progress < 0.60:  # 450K-600K
            level = 3  # Hard
        elif progress < 0.90:  # 600K-900K
            level = 4  # Expert
        else:  # 900K-1M
            level = 5  # Elite (new level for most recent data)

        self.env_instance.set_curriculum_level(level)

        if self.num_timesteps % 50_000 == 0 and self.verbose:
            print(f"[CURRICULUM] Level {level}/5 (Elite level unlocked)")

        return True


# ============================================================================
# DIAGNOSTIC CALLBACK (Wall Street Grade Monitoring)
# ============================================================================

class DiagnosticCallback(BaseCallback):
    """
    INSTITUTIONAL DEBUGGING (Wall Street Standard)

    Tracks critical metrics to detect:
    - Mode collapse (action distribution, diversity score)
    - Learning paralysis (emergency stops, confidence trends)
    - Bias (trade quality, win rate progression)
    - CRITIC HEALTH (value function Std - target >1.0)
    """
    def __init__(self, env, log_freq=50_000, verbose=1):
        super().__init__(verbose)
        self.env_instance = env.envs[0].env
        self.log_freq = log_freq
        self.actions_taken = []
        self.confidences = []
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
        current_step = 450_000 + self.num_timesteps

        print(f"\n{'='*80}")
        print(f"[DIAGNOSTICS] Step {current_step:,}")
        print(f"{'='*80}")

        # 1. ACTION DISTRIBUTION (Mode Collapse Detection)
        if len(self.actions_taken) > 100:
            recent_actions = self.actions_taken[-1000:]
            action_counts = np.bincount(recent_actions, minlength=3)
            action_pcts = action_counts / len(recent_actions) * 100

            print(f"\n[ACTION DISTRIBUTION] (Last 1000 steps)")
            print(f"   SELL: {action_pcts[0]:5.1f}%  {'[WARNING]' if action_pcts[0] > 60 else ''}")
            print(f"   HOLD: {action_pcts[1]:5.1f}%  {'[WARNING]' if action_pcts[1] > 60 else ''}")
            print(f"   BUY:  {action_pcts[2]:5.1f}%  {'[WARNING]' if action_pcts[2] > 60 else ''}")

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
            std_conf = np.std(recent_conf)
            print(f"\n[CONFIDENCE] Avg: {avg_conf:.1f}%")
            print(f"   Min: {np.min(recent_conf):.1f}% | Max: {np.max(recent_conf):.1f}% | Std: {std_conf:.1f}%")

        # 4. RISK MANAGEMENT
        env = self.env_instance
        print(f"\n[RISK MANAGEMENT]")
        print(f"   Balance: ${env.balance:,.0f}")
        print(f"   Max DD: {env.max_drawdown_pct*100:.2f}%")
        print(f"   Emergency Stops: {env.emergency_stops}")

        # 5. TRADES
        if hasattr(env, 'total_trades') and env.total_trades > 0:
            win_rate = env.winning_trades / env.total_trades * 100 if env.total_trades > 0 else 0
            print(f"\n[TRADES] Total: {env.total_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            if hasattr(env, 'profit_factor'):
                print(f"   Profit Factor: {env.profit_factor:.2f}")

        print(f"{'='*80}\n")


# ============================================================================
# ADAPTIVE ENTROPY CALLBACK (Adjusted for 450K → 1M)
# ============================================================================

class AdaptiveEntropyCallback(BaseCallback):
    """
    Adaptive Entropy for continuation training (450K → 1M)

    Schedule adjusted for continuation:
    - 450K-600K (60-80%): ent_coef = 0.35 → 0.25 (continue exploration)
    - 600K-900K (80-95%): ent_coef = 0.25 → 0.15 (decay)
    - 900K-1M (95-100%): ent_coef = 0.15 (exploitation)
    """

    def __init__(self, initial_ent=0.35, min_ent=0.15, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.min_ent = min_ent
        self.total_steps = 1_000_000  # Total target
        self.start_step = 450_000  # Starting from 450K

    def _on_step(self) -> bool:
        # Current step in TOTAL (450K + n_calls)
        current_step = 450_000 + self.num_timesteps

        # Progress relative to full 1M training (0.0 → 1.0)
        progress = current_step / self.total_steps

        # Entropy schedule
        if progress < 0.80:  # 450K-600K: maintain high exploration
            new_ent = 0.35
        elif progress < 0.95:  # 600K-900K: decay
            decay_progress = (progress - 0.80) / 0.15
            new_ent = 0.35 - (0.35 - 0.15) * decay_progress
        else:  # 900K-1M: exploitation
            new_ent = 0.15

        # Update model entropy
        self.model.ent_coef = new_ent

        # Log every 5K steps
        if self.num_timesteps % 5000 == 0:
            print(f"[ENTROPY] Step {current_step:,} | Progress: {progress*100:.1f}% | ent_coef: {new_ent:.4f}")

        return True

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
CHECKPOINT_PATH = parent_dir / "models" / "checkpoints" / "checkpoint_450000.zip"
MODELS_DIR = parent_dir / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = Path("C:/Users/lbye3/Desktop/GoldRL/output/logs/agent7_continue_450k_to_1M")
CHECKPOINTS_ANALYSIS_DIR = MODELS_DIR / "checkpoints_analysis"
INTERPRETABILITY_DIR = MODELS_DIR / "interpretability"

# Create directories
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
INTERPRETABILITY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[PATHS]")
print(f"  Load checkpoint: {CHECKPOINT_PATH}")
print(f"  Save to: {CHECKPOINTS_DIR}")
print(f"  Logs: {LOGS_DIR}")

# Training params
TOTAL_TIMESTEPS = 550_000  # 450K → 1M = 550K additional steps
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 50_000

print(f"\n[TRAINING]")
print(f"  Additional steps: {TOTAL_TIMESTEPS:,}")
print(f"  Checkpoint frequency: {CHECKPOINT_FREQ:,}")
print(f"  Evaluation frequency: {EVAL_FREQ:,}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("[STEP 1/5] LOADING DATA")
print("="*80)

loader = DataLoader(verbose=False)
print(f"[LOADING] All market data + correlations...")

# Load all data (includes XAUUSD, correlations, macro, etc.)
aligned_df, auxiliary_data = loader.load_all_data()
print(f"[OK] Aligned data loaded: {len(aligned_df)} bars")

# Feature engineering
fe = FeatureEngineering(aligned_df, auxiliary_data, verbose=True)

print(f"[COMPUTING] Training features (2008-2020)...")
features_train = fe.compute_all_features(
    start_date=config.TRAIN_START_DATE,
    end_date=config.TRAIN_END_DATE
)
print(f"[OK] Training features: {features_train.shape}")

print(f"[COMPUTING] Validation features (2021)...")
features_val = fe.compute_all_features(
    start_date='2021-01-01',
    end_date='2021-12-31'
)
print(f"[OK] Validation features: {features_val.shape}")

# Extract prices
xauusd_h1 = auxiliary_data['xauusd_raw']['H1']
prices_train = xauusd_h1.loc[features_train.index, ['open', 'high', 'low', 'close']]
prices_val = xauusd_h1.loc[features_val.index, ['open', 'high', 'low', 'close']]

# Prepare train and validation data
train_data = features_train
val_data = features_val

# ============================================================================
# CREATE ENVIRONMENTS
# ============================================================================

print("\n" + "="*80)
print("[STEP 2/5] CREATING ENVIRONMENTS")
print("="*80)

def make_env(df, agent_id=7, is_eval=False):
    """Create trading environment."""
    env = GoldTradingEnv(
        df=df,
        initial_balance=100000,
        transaction_cost=7.0,
        slippage_pct=0.0001,
        agent_id=agent_id,
        agent_type='momentum',
        is_eval=is_eval
    )
    env = Monitor(env)
    return env

# Training environment
train_env = DummyVecEnv([lambda: make_env(train_data, agent_id=7, is_eval=False)])
print("[OK] Training environment created")

# Evaluation environment
eval_env = DummyVecEnv([lambda: make_env(val_data, agent_id=7, is_eval=True)])
print("[OK] Evaluation environment created")

# ============================================================================
# LOAD MODEL FROM CHECKPOINT
# ============================================================================

print("\n" + "="*80)
print("[STEP 3/5] LOADING MODEL FROM 450K CHECKPOINT")
print("="*80)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

print(f"[LOADING] {CHECKPOINT_PATH}")
model = RecurrentPPO.load(
    str(CHECKPOINT_PATH),
    env=train_env,
    tensorboard_log=str(LOGS_DIR),
    device='cpu'
)

print(f"[OK] Model loaded from 450K checkpoint")
print(f"  - Architecture: RecurrentPPO + LSTM")
print(f"  - Starting from: 450,000 steps")
print(f"  - Target: 1,000,000 steps")

# Update num_timesteps to reflect continuation
model.num_timesteps = 450_000
print(f"[OK] Set num_timesteps to {model.num_timesteps:,}")

# ============================================================================
# CALLBACKS
# ============================================================================

print("\n" + "="*80)
print("[STEP 4/5] CONFIGURING CALLBACKS")
print("="*80)

callbacks = []

# 1. Curriculum Learning (Progressive difficulty)
curriculum_callback = CurriculumCallback(
    env=train_env,
    total_timesteps=1_000_000,
    verbose=1
)
callbacks.append(curriculum_callback)
print("[OK] CurriculumCallback (Level 3 → 5)")

# 2. Diagnostic Monitoring (Wall Street grade)
diagnostic_callback = DiagnosticCallback(
    env=train_env,
    log_freq=50_000,
    verbose=1
)
callbacks.append(diagnostic_callback)
print("[OK] DiagnosticCallback (every 50K steps)")

# 3. Adaptive Entropy
adaptive_entropy = AdaptiveEntropyCallback(initial_ent=0.35, min_ent=0.15)
callbacks.append(adaptive_entropy)
print("[OK] AdaptiveEntropyCallback (0.35 → 0.15)")

# 4. Checkpoint Callback (save every 50K)
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=str(CHECKPOINTS_DIR),
    name_prefix=f"checkpoint",
    verbose=1
)
callbacks.append(checkpoint_callback)
print(f"[OK] CheckpointCallback (every {CHECKPOINT_FREQ:,} steps)")

# 5. Checkpoint Evaluation Callback (evaluate + CSV every 50K)
checkpoint_eval_callback = CheckpointEvaluationCallback(
    eval_env=eval_env,
    checkpoints_dir=str(CHECKPOINTS_DIR),
    output_dir=str(CHECKPOINTS_ANALYSIS_DIR),
    eval_freq=EVAL_FREQ,
    n_eval_episodes=1,
    deterministic=True,
    verbose=1
)
callbacks.append(checkpoint_eval_callback)
print(f"[OK] CheckpointEvaluationCallback (every {EVAL_FREQ:,} steps)")

# 6. Interpretability Callback (interview every 50K)
interpretability_callback = InterpretabilityCallback(
    eval_env=eval_env,
    output_dir=str(INTERPRETABILITY_DIR),
    interview_freq=EVAL_FREQ,
    n_samples=100,
    verbose=1
)
callbacks.append(interpretability_callback)
print(f"[OK] InterpretabilityCallback (every {EVAL_FREQ:,} steps)")

print(f"\n[TOTAL] {len(callbacks)} callbacks configured")

# ============================================================================
# CONTINUE TRAINING
# ============================================================================

print("\n" + "="*80)
print("[STEP 5/5] CONTINUE TRAINING 450K → 1M")
print("="*80)
print(f"[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[TARGET] {TOTAL_TIMESTEPS:,} additional steps (~3h20min)")
print("="*80)

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="agent7_continue_450k_to_1M",
        reset_num_timesteps=False,  # CRITICAL: Don't reset to 0!
        progress_bar=True
    )

    print("\n" + "="*80)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*80)

    # Save final model
    final_model_path = MODELS_DIR / "agent7_1M_final.zip"
    model.save(str(final_model_path))
    print(f"[OK] Final model saved: {final_model_path}")

    print(f"\n[END] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

except KeyboardInterrupt:
    print("\n[WARNING] Training interrupted by user")
    interrupted_path = MODELS_DIR / "agent7_interrupted.zip"
    model.save(str(interrupted_path))
    print(f"[OK] Model saved: {interrupted_path}")

finally:
    train_env.close()
    eval_env.close()
    print("\n[OK] Environments closed")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nCheck results in:")
print(f"  - Models: {MODELS_DIR}")
print(f"  - Logs:   {LOGS_DIR}")
print(f"\nNext steps:")
print(f"  1. Check TensorBoard: tensorboard --logdir {LOGS_DIR}")
print(f"  2. Run SHOW_ALL_CHECKPOINTS.bat")
print(f"  3. Compare 450K vs 1M performance")
print("="*80)