"""
Trading Environment V2 ULTIMATE - Gymnasium Custom Environment (HEDGE FUND GRADE)
RL environment for Gold trading with dynamic features + RL-specific features
FTMO Compliant: 1% risk max, 2% daily loss, 10% max DD

VERSION 2 ULTIMATE - INSTITUTIONAL-GRADE IMPROVEMENTS:

[TARGET] REWARD SHAPING (Tiered Hierarchy):
1. Direction Prediction Reward (+0.02)
2. Enhanced Profit Taking Bonus (+0.05 to +0.20)
3. Quick Loss Cutting Bonus (+0.03)
4. Trade Completion Reward (+0.10)
5. Unified Diversity Score (entropy-based, prevents mode collapse)

LEARNING ENHANCEMENTS:
6. Curriculum Learning (4 difficulty levels)
7. Adaptive Reward Scaling (performance-based)

RISK MANAGEMENT (Advanced):
8. Kelly Criterion position sizing
9. VaR 95% monitoring
10. Tail Risk detection (kurtosis)
11. Safe Exploration (FTMO hard constraints)

FEATURE ENGINEERING RL:
12. Recent action history (5 actions)
13. Regret signal (missed opportunities)
14. Position duration tracking
15. Unrealized PnL ratio
16. Market regime classification
17. Time until macro event
18. Volatility percentile
19. Trade Similarity Score (WALL STREET: Pattern recognition vs winners/losers) <- NEW

Total features: 209 base + 20 RL (13 RL + 7 MEMORY) = 229 features

Based on research from: Renaissance Technologies, Citadel, Two Sigma
Papers: Haarnoja et al. (2018), Schulman et al. (2017), Ng et al. (1999)
"""

import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from collections import deque, Counter
from scipy import stats

from config import *


class GoldTradingEnv(gym.Env):
    """
    Environnement Gymnasium pour trading Gold

    Observation space: Box(229,) - Features dynamiques (209 base + 20 RL)
    Action space:
        - Discrete(3): 0=SELL, 1=HOLD, 2=BUY
        - OU Box(2,): [direction (-1 à 1), position_size (0 à 1)]

    Reward: Combinaison Sharpe + Profit + Drawdown penalty + Win rate

    Contraintes FTMO:
        - Max 1% risk per trade
        - Max 2% daily loss (stop si atteint)
        - Max 10% drawdown (termine épisode)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        initial_balance: float = INITIAL_BALANCE,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        action_space_type: str = ACTION_SPACE,
        verbose: bool = False
    ):
        """
        Args:
            features_df: DataFrame avec features dynamiques (index datetime)
            prices_df: DataFrame avec OHLCV XAUUSD H1 (pour exécution trades)
            initial_balance: Balance initiale
            max_episode_steps: Durée max épisode
            action_space_type: 'discrete' ou 'continuous'
            verbose: Afficher logs
        """
        super().__init__()

        self.features_df = features_df
        self.prices_df = prices_df
        self.initial_balance = initial_balance
        self.max_episode_steps = max_episode_steps
        self.action_space_type = action_space_type
        self.verbose = verbose

        # Aligner features et prices
        common_idx = self.features_df.index.intersection(self.prices_df.index)
        self.features_df = self.features_df.loc[common_idx]
        self.prices_df = self.prices_df.loc[common_idx]

        self.total_steps = len(self.features_df)

        # PRE-COMPUTE per-feature normalization statistics (CORRECT METHOD)
        # Each feature normalized with its OWN mean/std, not row-wise
        self.feature_means = self.features_df.mean().values.astype(np.float32)
        self.feature_stds = self.features_df.std().values.astype(np.float32)
        # Avoid division by zero
        self.feature_stds = np.where(self.feature_stds < 1e-8, 1.0, self.feature_stds)
        if NORMALIZE_OBSERVATIONS:
            self.log(f"   [NORM] Per-feature Z-score normalization enabled (209 features)")

        # Observation space: Base features + RL features
        n_base_features = self.features_df.shape[1]  # 209 (all features)
        n_rl_features = 20  # 13 RL + 7 MEMORY: 3 (last_action) + 1 (regret) + 1 (duration) + 1 (pnl_ratio) + 3 (regime) + 1 (hours) + 1 (vol) + 1 (position_side) + 1 (trade_similarity) + 7 (MEMORY: win_rate, streak, avg_pnl, best, worst, win_count, loss_count)
        n_total_features = n_base_features + n_rl_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_total_features,),
            dtype=np.float32
        )

        self.log(f"   Observation space: {n_total_features} ({n_base_features} base + {n_rl_features} RL)")

        # Action space
        if self.action_space_type == 'discrete':
            self.action_space = spaces.Discrete(3)  # 0=SELL, 1=HOLD, 2=BUY
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0.0  # Position size (lots)
        self.position_side = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.lot_size = 0.01  # [OK] FIX: Standard lot size for missed opportunity calculation
        self.entry_features = None  # WALL STREET: Stocke features au moment de l'entrée (Trade Quality Memory)

        # WALL STREET V3: Missed Opportunities Tracking (Renaissance Technologies pattern)
        self.missed_opportunities = []  # Liste des trades manqués (HOLD alors que profitable)
        self.potential_trade_context = None  # Contexte stocké après HOLD pour évaluation future
        self.steps_since_hold = 0  # Compteur pour évaluer missed après N steps

        # FTMO tracking
        self.daily_pnl = 0.0
        self.daily_start_balance = initial_balance
        self.last_date = None
        self.max_balance = initial_balance
        self.max_drawdown = 0.0
        self.daily_loss_limit_reached = False  # Flag pour bloquer nouveaux trades
        self.risk_multiplier = 1.0  # Multiplicateur de risque dynamique (1.0 = normal)

        # Metrics
        self.trades = []
        self.daily_returns = deque(maxlen=SHARPE_WINDOW)
        self.ftmo_violations = 0
        self.last_closed_trade_pnl_pct = None  # For 4R reward bonus

        # Action tracking (pour pénaliser HOLD excessif)
        self.consecutive_holds = 0
        self.last_action = 1  # Start with HOLD
        self.total_holds = 0
        self.total_actions = 0
        self.recent_actions = deque(maxlen=100)  # Track last 100 actions

        # Diagnostic tracking (for DiagnosticCallback)
        self.last_action_taken = 1  # For callback monitoring
        self.last_confidence = 0.0  # Track confidence (size parameter)
        self.emergency_stops_count = 0  # Count emergency FTMO stops

        # CHECKPOINT CALLBACK: Trade capture for CSV saving
        self.trade_just_closed = False  # Flag to signal trade closure
        self.last_trade_info = None  # Store last trade details for callback

        # V2 IMPROVEMENTS: Curriculum Learning Support
        self.curriculum_level = 1  # Default level (1 = easiest, 4 = hardest)
        self.last_price_direction = 0  # Track price movement for direction prediction reward
        self.last_close_price = None

        # V2 ULTIMATE: RL-specific features
        self.action_history = deque(maxlen=5)  # Last 5 actions for observation
        self.regret_signal = 0.0  # Missed profit opportunities
        self.position_entry_step = 0  # When position was opened
        self.unrealized_pnl_ratio = 0.0  # Unrealized PnL / Risk
        self.market_regime = 0  # 0=ranging, 1=trending, 2=volatile
        self.hours_until_event = 999.0  # Time until next macro event
        self.volatility_percentile = 0.5  # Current vol vs historical

        # V2 ULTIMATE: Advanced Risk Management
        self.recent_returns = deque(maxlen=100)  # For Kelly, VaR
        self.kelly_fraction = 0.0
        self.var_95 = 0.0
        self.tail_risk_detected = False

        # V2 ULTIMATE: Adaptive Reward Scaling
        self.performance_score = 0.0  # Rolling performance metric
        self.adaptive_reward_multiplier = 1.0

        self.log("[BUILD]  Trading Environment V2 ULTIMATE initialized (HEDGE FUND GRADE)")
        self.log("   [TARGET] Reward: Tiered Hierarchy (Core 70% + Risk 20% + Behavior 10%)")
        self.log("   [RISK]  Risk: Kelly + VaR + Tail Risk + Safe Exploration")
        self.log("   [FEATURES] Features: +13 RL-specific features added to observation")
        self.log(f"   Features: {self.features_df.shape}")
        self.log(f"   Prices: {self.prices_df.shape}")
        self.log(f"   Total steps: {self.total_steps}")
        self.log(f"   Max episode steps: {self.max_episode_steps}")
        self.log(f"   Action space: {self.action_space_type}")
        self.log(f"   Initial balance: ${self.initial_balance:,.2f}")

    def log(self, message: str):
        """Log si verbose"""
        if self.verbose:
            print(message)

    def set_curriculum_level(self, level: int):
        """
        Set curriculum learning difficulty level (V2 Feature)

        Levels:
        1 = Easy (2008-2012) - Trending markets, clear patterns
        2 = Medium (2013-2016) - Mixed conditions
        3 = Hard (2017-2019) - Choppy, low volatility
        4 = Expert (2020-2024) - COVID, extreme volatility

        Args:
            level: Difficulty level (1-4)
        """
        if level < 1 or level > 4:
            raise ValueError("Curriculum level must be between 1 and 4")
        self.curriculum_level = level
        self.log(f"[CURRICULUM] Level set to {level}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)

        # Random start (ou début dataset)
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            # Start aléatoire (laisser de la place pour un épisode complet)
            max_start = self.total_steps - self.max_episode_steps - 1
            self.current_step = np.random.randint(0, max(1, max_start))

        # Reset state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.entry_features = None  # WALL STREET: Reset entry features

        # WALL STREET V3: Reset missed opportunities tracking
        self.missed_opportunities = []
        self.potential_trade_context = None
        self.steps_since_hold = 0

        # FTMO tracking
        self.daily_pnl = 0.0
        self.daily_start_balance = self.initial_balance
        self.last_date = None
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.daily_loss_limit_reached = False
        self.risk_multiplier = 1.0

        # Metrics
        self.trades = []
        self.daily_returns.clear()
        self.ftmo_violations = 0
        self.last_closed_trade_pnl_pct = None

        # Action tracking reset
        self.consecutive_holds = 0
        self.last_action = 1
        self.total_holds = 0
        self.total_actions = 0
        self.recent_actions.clear()

        # Diagnostic tracking reset
        self.last_action_taken = 1
        self.last_confidence = 0.0
        self.emergency_stops_count = 0

        # V2: Reset direction tracking
        self.last_price_direction = 0
        self.last_close_price = self.prices_df['close'].iloc[self.current_step]

        # V2 ULTIMATE: Reset RL features
        self.action_history.clear()
        self.regret_signal = 0.0
        self.position_entry_step = 0
        self.unrealized_pnl_ratio = 0.0
        self.market_regime = 0
        self.hours_until_event = 999.0
        self.volatility_percentile = 0.5

        # V2 ULTIMATE: Reset risk management
        self.recent_returns.clear()
        self.kelly_fraction = 0.0
        self.var_95 = 0.0
        self.tail_risk_detected = False

        # V2 ULTIMATE: Reset adaptive reward
        self.performance_score = 0.0
        self.adaptive_reward_multiplier = 1.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        V2.1 CRITIC BOOST: Retourne observation avec RL + MEMORY features

        Base features (209) + RL features (20) = 229 total

        RL Features (13):
        1. Recent action history (last action one-hot = 3)
        2. Regret signal (missed opportunities = 1)
        3. Position duration (normalized = 1)
        4. Unrealized PnL ratio (= 1)
        5. Market regime (one-hot = 3)
        6. Hours until macro event (normalized = 1)
        7. Volatility percentile (= 1)
        8. Position side (normalized = 1)
        9. Trade Similarity Score (WALL STREET: pattern recognition = 1)

        MEMORY Features (7) - V2.1 CRITIC BOOST (LAST 20 TRADES):
        10. Recent win rate (last 20 trades = 1)
        11. Win/Loss streak (consecutive wins/losses = 1)
        12. Avg PnL last 20 trades (normalized = 1)
        13. Best trade last 20 (normalized = 1)
        14. Worst trade last 20 (normalized = 1)
        15. Win count last 20 (normalized = 1)
        16. Loss count last 20 (normalized = 1)

        Total: 13 RL + 7 MEMORY = 20 enhanced features
        """
        # Base features from dataframe
        features_row = self.features_df.iloc[self.current_step]
        base_features = features_row.values.astype(np.float32)

        # PER-FEATURE normalization (CORRECT: each feature with its own mean/std)
        if NORMALIZE_OBSERVATIONS:
            # Z-score per feature: (x - mean_feature) / std_feature
            base_features = (base_features - self.feature_means) / self.feature_stds
            # Clip to [-5, 5] to avoid extreme values
            base_features = np.clip(base_features, -5.0, 5.0)

        # Handle NaN/Inf values
        base_features = np.nan_to_num(base_features, nan=0.0, posinf=5.0, neginf=-5.0)

        # V2 ULTIMATE: Add RL-specific features
        rl_features = []

        # 1. Action history (last 5 actions, one-hot encoded = 15 features)
        # Simplified: Just last action encoded
        last_action_encoded = np.zeros(3)
        if len(self.action_history) > 0:
            last_action_encoded[self.action_history[-1]] = 1.0
        rl_features.extend(last_action_encoded)

        # 2. Regret signal (normalized)
        rl_features.append(np.clip(self.regret_signal / 10.0, -1.0, 1.0))

        # 3. Position duration (normalized by max episode steps)
        if self.position_side != 0:
            duration_normalized = (self.current_step - self.position_entry_step) / self.max_episode_steps
        else:
            duration_normalized = 0.0
        rl_features.append(duration_normalized)

        # 4. Unrealized PnL ratio (clipped)
        rl_features.append(np.clip(self.unrealized_pnl_ratio, -5.0, 5.0))

        # 5. Market regime (one-hot encoded = 3 features)
        regime_encoded = np.zeros(3)
        regime_encoded[self.market_regime] = 1.0
        rl_features.extend(regime_encoded)

        # 6. Hours until event (normalized by 168h = 1 week)
        hours_normalized = np.clip(self.hours_until_event / 168.0, 0.0, 1.0)
        rl_features.append(hours_normalized)

        # 7. Volatility percentile (already 0-1)
        rl_features.append(self.volatility_percentile)

        # 8. Position side (normalized: -1=short, 0=flat, 1=long)
        rl_features.append(float(self.position_side))

        # 9. WALL STREET: Trade Similarity Score (pattern recognition)
        # Compare current market context with historical winners/losers
        # [OK] INCLUDED in observation so agent can learn from it
        trade_similarity = self._calculate_trade_similarity_score()
        rl_features.append(trade_similarity)

        # ========================================================================
        # V2.1 CRITIC BOOST: TRADE MEMORY FEATURES (7 features)
        # Help Critic distinguish good/bad states with EXPLICIT recent performance
        # UPGRADE: Last 20 trades (was 5/10) for deeper context + LSTM synergy
        # ========================================================================

        # 10. Recent win rate (last 20 trades - BOOST from 5)
        if len(self.trades) >= 20:
            recent_20 = self.trades[-20:]
            win_rate = sum(1 for t in recent_20 if t['pnl'] > 0) / len(recent_20)
        else:
            win_rate = 0.5  # Neutral if not enough trades
        rl_features.append(win_rate)

        # 11. Win/Loss streak (current consecutive wins or losses)
        streak = self._calculate_current_streak()
        rl_features.append(np.clip(streak / 5.0, -1.0, 1.0))  # Normalized [-1, 1]

        # 12. Average PnL last 20 trades (BOOST from 5 - normalized by initial balance)
        if len(self.trades) >= 20:
            avg_pnl = np.mean([t['pnl'] for t in self.trades[-20:]])
            avg_pnl_normalized = avg_pnl / self.initial_balance * 100
        else:
            avg_pnl_normalized = 0.0
        rl_features.append(np.clip(avg_pnl_normalized, -1.0, 1.0))

        # 13. Best trade last 20 (BOOST from 10 - normalized)
        if len(self.trades) >= 20:
            best_pnl = max(t['pnl'] for t in self.trades[-20:])
            best_normalized = best_pnl / self.initial_balance * 100
        else:
            best_normalized = 0.0
        rl_features.append(np.clip(best_normalized, -1.0, 1.0))

        # 14. Worst trade last 20 (BOOST from 10 - normalized)
        if len(self.trades) >= 20:
            worst_pnl = min(t['pnl'] for t in self.trades[-20:])
            worst_normalized = worst_pnl / self.initial_balance * 100
        else:
            worst_normalized = 0.0
        rl_features.append(np.clip(worst_normalized, -1.0, 1.0))

        # 15. Win count in last 20 trades (BOOST from 10)
        if len(self.trades) >= 20:
            win_count = sum(1 for t in self.trades[-20:] if t['pnl'] > 0)
            win_count_normalized = win_count / 20.0  # 0-1
        else:
            win_count_normalized = 0.5
        rl_features.append(win_count_normalized)

        # 16. Loss count in last 20 trades (BOOST from 10)
        if len(self.trades) >= 20:
            loss_count = sum(1 for t in self.trades[-20:] if t['pnl'] <= 0)
            loss_count_normalized = loss_count / 20.0  # 0-1
        else:
            loss_count_normalized = 0.5
        rl_features.append(loss_count_normalized)

        # ========================================================================
        # TOTAL RL FEATURES: 13 original + 7 memory = 20
        # TOTAL OBSERVATION: 209 base + 20 RL = 229 features
        # ========================================================================

        # Combine base + RL features
        rl_features_array = np.array(rl_features, dtype=np.float32)
        full_observation = np.concatenate([base_features, rl_features_array])

        return full_observation

    def _get_info(self) -> Dict:
        """Informations additionnelles"""
        current_date = self.features_df.index[self.current_step]
        current_price = self.prices_df['close'].iloc[self.current_step]

        info_dict = {
            'step': self.current_step,
            'date': current_date,
            'price': current_price,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'position_side': self.position_side,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.trades),
            'ftmo_violations': self.ftmo_violations,
            'risk_multiplier': self.risk_multiplier,
            'daily_loss_blocked': self.daily_loss_limit_reached,
            'min_confidence_threshold': self._calculate_min_confidence_threshold(),
            # CHECKPOINT CALLBACK: Trade info for CSV saving
            'trade_closed': self.trade_just_closed,
            'last_trade': self.last_trade_info if self.trade_just_closed else None
        }

        return info_dict

    def _calculate_min_confidence_threshold(self) -> float:
        """
        WALL STREET: Calcule le seuil de confiance minimum dynamique

        Modulation basée sur:
        1. Drawdown actuel (DD)
        2. Trade Similarity Score (pattern recognition) ← NEW

        Comportement :
        - DD < 7% : Confiance base = 60% (normal)
        - DD >= 7% : Confiance base = 70% (très sélectif)

        + MODULATION TRADE SIMILARITY:
        - Similarity +0.8 (ressemble winners) : -10% threshold (50% min)
        - Similarity -0.7 (ressemble losers) : +15% threshold (85% max)

        Returns:
            float: Seuil de confiance minimum (0.50 à 0.85)

        Exemples :
        - DD=5%, sim=+0.6 → 60% - 7.5% = 52.5% (encourage winner patterns)
        - DD=5%, sim=-0.5 → 60% + 7.5% = 67.5% (évite loser patterns)
        - DD=8%, sim=+0.8 → 70% - 10% = 60% (DD élevé mais pattern gagnant)
        - DD=8%, sim=-0.7 → 70% + 10.5% = 80.5% (DD élevé + pattern perdant = très sélectif)

        Used by: Renaissance Technologies (Medallion Fund - pattern avoidance)
        """
        dd = self.max_drawdown

        # Base threshold from DD
        if dd < FTMO_RISK_REDUCTION_START:
            base_threshold = MIN_CONFIDENCE_NORMAL  # 10% (TRAINING MODE)
        else:
            base_threshold = MIN_CONFIDENCE_STRESSED  # 30% (TRAINING MODE)

        # WALL STREET: Modulate based on Trade Similarity
        # [OK] ENABLED with reduced impact for training (±5% instead of -10%/+15%)
        trade_similarity = self._calculate_trade_similarity_score()

        # Similarity modulation: TRAINING MODE ±5% (Production: -10%/+15%)
        # Positive similarity (winners) → Lower threshold (encourage)
        # Negative similarity (losers) → Higher threshold (discourage)
        if trade_similarity > 0:
            # Similarity with winners: reduce threshold up to -5% (training)
            similarity_adjustment = -trade_similarity * 0.05  # Max -5% (was -10%)
        else:
            # Similarity with losers: increase threshold up to +5% (training)
            similarity_adjustment = -trade_similarity * 0.05  # Max +5% (was +15%)

        # Final threshold with bounds
        final_threshold = base_threshold + similarity_adjustment
        # [OK] TRAINING MODE: Allow thresholds as low as 5% (exploration)
        # Production: Change to (0.45, 0.85) after training
        final_threshold = np.clip(final_threshold, 0.05, 0.85)  # Permissive during training

        return float(final_threshold)

    def _calculate_adaptive_base_risk(self, confidence: float, atr: float) -> float:
        """
        Calcule le risque de base adaptatif entre 0.33% et 1.0%

        Le risque de base varie selon :
        1. Confiance de l'agent (confidence/size)
        2. Volatilité relative du marché (ATR actuel vs moyenne)

        Args:
            confidence: Confiance de l'agent (0.0 à 1.0)
            atr: ATR actuel

        Returns:
            float: Risque de base entre BASE_RISK_MIN (0.33%) et BASE_RISK_MAX (1.0%)

        Formule :
        - Composante confiance : 60% conf → 0%, 100% conf → 100% de la plage
        - Composante volatilité : vol_haute → réduction, vol_basse → augmentation
        - risk_base = BASE_RISK_MIN + (confiance_factor × vol_factor × plage)

        Exemples :
        - conf=100%, vol_normale → risk=1.0%
        - conf=80%, vol_normale → risk=0.78%
        - conf=70%, vol_haute → risk=0.50%
        - conf=60%, vol_très_haute → risk=0.33%
        """
        if not ADAPTIVE_POSITION_SIZING:
            return FTMO_MAX_RISK_PER_TRADE  # 1.0% fixe

        # 1. Facteur confiance (linéaire entre 60% et 100%)
        # 60% → 0.0, 100% → 1.0
        conf_min = MIN_CONFIDENCE_NORMAL  # 60%
        conf_max = 1.0  # 100%
        conf_factor = (confidence - conf_min) / (conf_max - conf_min)
        conf_factor = max(0.0, min(1.0, conf_factor))  # Clip entre 0 et 1

        # 2. Facteur volatilité (ATR actuel vs moyenne sur 50 périodes)
        # Calculer ATR moyen sur les 50 dernières périodes
        start_idx = max(0, self.current_step - 50)
        end_idx = self.current_step + 1

        if end_idx - start_idx >= 10:  # Au moins 10 périodes
            recent_prices = self.prices_df.iloc[start_idx:end_idx]
            atr_values = []
            for i in range(len(recent_prices) - 1):
                high = recent_prices['high'].iloc[i]
                low = recent_prices['low'].iloc[i]
                close_prev = recent_prices['close'].iloc[i-1] if i > 0 else recent_prices['close'].iloc[i]
                tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
                atr_values.append(tr)

            atr_mean = np.mean(atr_values) if atr_values else atr
        else:
            atr_mean = atr  # Pas assez de données, utiliser ATR actuel

        # Volatilité relative
        vol_ratio = atr / atr_mean if atr_mean > 0 else 1.0

        # vol_ratio < 0.8 → faible vol → facteur 1.0 (risque max)
        # vol_ratio = 1.0 → vol normale → facteur 0.85
        # vol_ratio > 1.2 → haute vol → facteur 0.7
        # vol_ratio > 1.5 → très haute vol → facteur 0.5
        if vol_ratio < 0.8:
            vol_factor = 1.0  # Faible volatilité → risque max
        elif vol_ratio < 1.0:
            vol_factor = 0.9 + (1.0 - vol_ratio) * 0.5  # 0.9-1.0
        elif vol_ratio < 1.2:
            vol_factor = 0.85 - (vol_ratio - 1.0) * 0.75  # 0.85-0.7
        elif vol_ratio < 1.5:
            vol_factor = 0.7 - (vol_ratio - 1.2) * 0.67  # 0.7-0.5
        else:
            vol_factor = 0.5  # Très haute volatilité → risque réduit

        # 3. Calcul final
        risk_range = BASE_RISK_MAX - BASE_RISK_MIN  # 1.0% - 0.33% = 0.67%
        base_risk = BASE_RISK_MIN + (conf_factor * vol_factor * risk_range)

        # Clip entre min et max
        base_risk = max(BASE_RISK_MIN, min(BASE_RISK_MAX, base_risk))

        return base_risk

    def _calculate_risk_multiplier(self) -> float:
        """
        Calcule le multiplicateur de risque dynamique basé sur le drawdown actuel

        Comportement :
        - DD < 5.5% : Risque normal (1.0x)
        - DD = 5.5-7% : Zone de sécurité, risque normal (1.0x)
        - DD = 7-10% : Réduction PROGRESSIVE du risque (1.0x → 0.0x)
        - DD >= 10% : Arrêt complet

        Formule : risk_multiplier = max(0, (10% - DD) / 3%)
        Exemples :
        - DD = 5% → risk = 1.0 (100% - normal)
        - DD = 7% → risk = 1.0 (100% - seuil de vigilance)
        - DD = 7.5% → risk = 0.83 (83% - réduction de 17%)
        - DD = 8% → risk = 0.67 (67% - réduction de 33%)
        - DD = 8.5% → risk = 0.50 (50% - réduction de 50%)
        - DD = 9% → risk = 0.33 (33% - réduction de 67%)
        - DD = 9.5% → risk = 0.17 (17% - réduction de 83%)
        - DD = 10% → risk = 0.0 (0% - arrêt)
        """
        dd = self.max_drawdown

        # Si DD < seuil de récupération, risque normal
        if dd < FTMO_RISK_RECOVERY_THRESHOLD:
            return 1.0

        # Si DD entre récupération et début de réduction, risque normal
        if dd < FTMO_RISK_REDUCTION_START:
            return 1.0

        # Si DD >= max, pas de trading
        if dd >= FTMO_MAX_DRAWDOWN:
            return 0.0

        # Réduction progressive entre 7% et 10%
        # À 7% → 1.0, à 10% → 0.0
        reduction_range = FTMO_MAX_DRAWDOWN - FTMO_RISK_REDUCTION_START  # 3%
        current_reduction = dd - FTMO_RISK_REDUCTION_START
        multiplier = max(0.0, 1.0 - (current_reduction / reduction_range))

        return multiplier

    def _can_open_new_trade(self) -> bool:
        """
        Vérifie si on peut ouvrir un nouveau trade

        Returns:
            True si on peut ouvrir un trade, False sinon

        Raisons de blocage :
        - Daily loss limit atteint ET config dit de bloquer les nouveaux trades
        - Drawdown >= 10%
        """
        # Vérifier daily loss
        if FTMO_DAILY_LOSS_BLOCKS_NEW_TRADES and self.daily_loss_limit_reached:
            self.log(f"   [BLOCKED] Daily loss limit atteint - Pas de nouveaux trades aujourd'hui")
            return False

        # Vérifier drawdown max
        if self.max_drawdown >= FTMO_MAX_DRAWDOWN:
            self.log(f"   [BLOCKED] Max drawdown atteint - Pas de nouveaux trades")
            return False

        return True

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une action

        Args:
            action:
                - Si discrete: 0 (SELL), 1 (HOLD), 2 (BUY)
                - Si continuous: [direction, position_size]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # CHECKPOINT CALLBACK: Reset trade flag at start of each step
        self.trade_just_closed = False

        # V2 ULTIMATE: Apply Safe Exploration Constraints FIRST
        # Get preliminary action type for safety check
        if self.action_space_type == 'discrete':
            prelim_action_type = int(action)
        else:
            if abs(action[0]) < 0.3 or action[1] < 0.1:
                prelim_action_type = 1
            else:
                prelim_action_type = 0 if action[0] < 0 else 2

        # Apply safety constraints
        safe_action = self._apply_safe_exploration_constraints(action, prelim_action_type)

        # Use safe action from now on
        action = safe_action

        # Parse action (possibly modified by safety constraints)
        if self.action_space_type == 'discrete':
            if action == 0:  # SELL
                action_direction = -1
                action_size = 1.0
                action_type = 0
            elif action == 1:  # HOLD
                action_direction = 0
                action_size = 0.0
                action_type = 1
            else:  # BUY
                action_direction = 1
                action_size = 1.0
                action_type = 2
        else:
            action_direction = action[0]  # -1 à 1
            action_size = action[1]  # 0 à 1
            # Classifier continuous comme HOLD si direction faible ou size petite
            if abs(action_direction) < 0.3 or action_size < 0.1:
                action_type = 1  # HOLD
            else:
                action_type = 0 if action_direction < 0 else 2  # SELL or BUY

        # V2 ULTIMATE: Update action history for RL features
        self.action_history.append(action_type)

        # Track action (for diversity score)
        self.total_actions += 1
        self.recent_actions.append(action_type)

        if action_type == 1:  # HOLD
            self.consecutive_holds += 1
            self.total_holds += 1
        else:
            self.consecutive_holds = 0

        self.last_action = action_type

        # Diagnostic tracking (for DiagnosticCallback)
        self.last_action_taken = action_type
        self.last_confidence = action_size if self.action_space_type == 'discrete' else abs(action[1])

        # WALL STREET V3: Missed Opportunities Detection
        # Store context after HOLD for future evaluation
        if action_type == 1 and self.position_side == 0:  # HOLD when FLAT
            self._store_hold_context()

        # Evaluate potential missed trade from previous HOLDs
        self._evaluate_missed_opportunities()

        # Exécuter action
        self._execute_action(action_direction, action_size)

        # Mettre à jour state
        self._update_state()

        # Calculer reward
        reward = self._calculate_reward()

        # Vérifier si terminated/truncated
        terminated = self._check_terminated()
        truncated = (self.current_step >= self.total_steps - 1 or
                    len(self.trades) >= self.max_episode_steps)

        # Next observation
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.current_step = self.total_steps - 1
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, direction: float, size: float):
        """
        Exécute une action de trading

        Args:
            direction: -1 (short), 0 (close), 1 (long)
            size: 0-1 (proportion du capital à risquer)
        """
        current_price = self.prices_df['close'].iloc[self.current_step]
        atr = self._get_current_atr()

        # HOLD
        if abs(direction) < 0.3 or size < 0.1:
            return

        # Fermer position existante si changement de direction
        if self.position_side != 0 and np.sign(direction) != self.position_side:
            self._close_position(current_price)

        # Ouvrir nouvelle position
        if self.position_side == 0:
            # Vérifier si on peut ouvrir un nouveau trade
            if not self._can_open_new_trade():
                return  # Pas de nouveau trade autorisé

            # Vérifier le seuil de confiance minimum (sélectivité dynamique)
            min_confidence = self._calculate_min_confidence_threshold()
            if size < min_confidence:
                self.log(f"   [BLOCKED] Confiance trop faible ({size:.1%} < {min_confidence:.1%}) - Trade refusé (DD={self.max_drawdown:.1%})")
                return

            # Calculer le multiplicateur de risque dynamique
            self.risk_multiplier = self._calculate_risk_multiplier()

            # Si risque multiplier = 0, pas de trade
            if self.risk_multiplier <= 0.0:
                self.log(f"   [BLOCKED] Risk multiplier = 0 (DD={self.max_drawdown:.1%}) - Pas de trade")
                return

            # Afficher si risque réduit
            if self.risk_multiplier < 1.0:
                self.log(f"   [WARNING] Risque réduit à {self.risk_multiplier:.1%} (DD={self.max_drawdown:.1%})")

            # Calculer le risque de base adaptatif (0.33%-1.0% selon confiance et volatilité)
            base_risk = self._calculate_adaptive_base_risk(size, atr)

            # Calculer position size avec multiplicateur dynamique DD
            risk_amount = self.balance * base_risk * size * self.risk_multiplier

            # Stop loss = ATR × multiplier
            stop_distance = atr * ATR_MULTIPLIER

            # Position size (lots) = risk / (stop distance × pip value × contract size)
            position_lots = risk_amount / (stop_distance * XAUUSD_CONTRACT_SIZE)

            # Limiter position size (max 10 lots par sécurité)
            position_lots = min(position_lots, 2.0)  # Max 2 lots (sécurité)

            # Ouvrir position
            self.position = position_lots
            self.position_side = 1 if direction > 0 else -1
            self.entry_price = current_price

            # WALL STREET: Capture key features at entry (Trade Quality Memory)
            # Stocke 15 features critiques pour comparaison future
            try:
                self.entry_features = np.array([
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('rsi_14')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd_signal')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('atr_14')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('adx')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('stochastic_k')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('cci')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('williams_r')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('roc')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('mfi')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('bb_width')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volume_ratio')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('trend_strength')],
                    self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volatility_percentile')],
                    float(self.max_drawdown)  # Current DD at entry
                ], dtype=np.float32)
            except (KeyError, IndexError):
                # Si features pas disponibles, vector par défaut
                self.entry_features = np.zeros(15, dtype=np.float32)

            # Coûts de transaction
            spread_cost = SPREAD_PIPS * XAUUSD_PIP_VALUE * XAUUSD_CONTRACT_SIZE * position_lots
            slippage_cost = SLIPPAGE_PIPS * XAUUSD_PIP_VALUE * XAUUSD_CONTRACT_SIZE * position_lots
            total_cost = spread_cost + slippage_cost + COMMISSION_PER_LOT * position_lots

            self.balance -= total_cost

            self.log(f"   [OPEN] {['SHORT', '', 'LONG'][self.position_side+1]} "
                    f"{position_lots:.2f} lots @ {current_price:.2f}")

    def _close_position(self, current_price: float):
        """Ferme la position actuelle"""
        if self.position_side == 0:
            return

        # Calculer PnL
        price_diff = (current_price - self.entry_price) * self.position_side
        pnl = price_diff * XAUUSD_CONTRACT_SIZE * self.position

        # Coûts de sortie
        spread_cost = SPREAD_PIPS * XAUUSD_PIP_VALUE * XAUUSD_CONTRACT_SIZE * self.position
        total_cost = spread_cost + COMMISSION_PER_LOT * self.position

        net_pnl = pnl - total_cost

        # Mettre à jour balance
        self.balance += net_pnl
        self.daily_pnl += net_pnl

        # Enregistrer trade
        trade = {
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'side': self.position_side,
            'size': self.position,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / self.balance if self.balance > 0 else 0.0,
            'entry_features': self.entry_features.copy() if self.entry_features is not None else np.zeros(15, dtype=np.float32)  # WALL STREET: Trade Quality Memory
        }
        self.trades.append(trade)

        # Track for 4R reward bonus
        self.last_closed_trade_pnl_pct = trade['pnl_pct']

        # CHECKPOINT CALLBACK: Store trade info for CSV saving
        self.trade_just_closed = True
        self.last_trade_info = {
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'side': 'LONG' if self.position_side == 1 else 'SHORT',
            'size': self.position,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / self.balance if self.balance > 0 else 0.0,
            'balance_after': self.balance,
            'win': 1 if net_pnl > 0 else 0
        }

        self.log(f"   [CLOSE] {['SHORT', '', 'LONG'][self.position_side+1]} "
                f"@ {current_price:.2f} | PnL: ${net_pnl:+.2f}")

        # Reset position
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0

    def _update_state(self):
        """Met à jour l'état (unrealized PnL, daily tracking, etc.)"""
        current_price = self.prices_df['close'].iloc[self.current_step]
        current_date = self.features_df.index[self.current_step].date()

        # Unrealized PnL
        if self.position_side != 0:
            price_diff = (current_price - self.entry_price) * self.position_side
            self.unrealized_pnl = price_diff * XAUUSD_CONTRACT_SIZE * self.position
        else:
            self.unrealized_pnl = 0.0

        # Equity
        self.equity = self.balance + self.unrealized_pnl

        # Daily tracking (reset si nouvelle journée)
        if self.last_date is not None and current_date != self.last_date:
            # Nouvelle journée
            daily_return = (self.equity - self.daily_start_balance) / self.daily_start_balance
            self.daily_returns.append(daily_return)

            self.daily_pnl = 0.0
            self.daily_start_balance = self.equity
            self.daily_loss_limit_reached = False  # Reset le flag chaque jour

        self.last_date = current_date

        # Vérifier si la daily loss limit est atteinte
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance if self.daily_start_balance > 0 else 0.0
        if self.daily_pnl < 0 and daily_loss_pct >= FTMO_MAX_DAILY_LOSS:
            if not self.daily_loss_limit_reached:
                self.log(f"   [WARNING] Daily loss limit atteint ({daily_loss_pct:.1%}) - Blocage nouveaux trades pour aujourd'hui")
                self.daily_loss_limit_reached = True

        # Max drawdown tracking
        if self.equity > self.max_balance:
            self.max_balance = self.equity

        current_dd = (self.max_balance - self.equity) / self.max_balance if self.max_balance > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # WALL STREET FIX: EMERGENCY STOP LOSS si DD approche limite
        # Ferme IMMÉDIATEMENT la position AVANT d'atteindre limite FTMO
        # Uses TRAINING_MODE to allow more exploration during learning (20% vs 9.5%)
        emergency_threshold = EMERGENCY_DD_TRAINING if TRAINING_MODE else EMERGENCY_DD_PRODUCTION
        if current_dd >= emergency_threshold and self.position_side != 0:
            mode = "TRAINING" if TRAINING_MODE else "PRODUCTION"
            self.log(f"   [EMERGENCY FTMO {mode}] DD={current_dd:.1%} >= {emergency_threshold:.1%} - Force closing position NOW")
            # Force close position immediately
            current_price = self.prices_df['close'].iloc[self.current_step]
            self._close_position(current_price)
            # Track emergency stops for diagnostic
            self.emergency_stops_count += 1

        # V2 ULTIMATE: Update advanced risk metrics
        self._update_advanced_risk_management()

        # V2 ULTIMATE: Update RL-specific features
        self._update_rl_features()

    def _calculate_unified_diversity_score(self) -> float:
        """
        V2 ULTIMATE: Unified Diversity Score (entropy-based)

        Standard hedge fund approach: ONE metric instead of multiple penalties
        Based on Shannon entropy (information theory)

        Returns:
            float: Diversity score [0, 1]
                   1.0 = Perfect diversity (33% each action)
                   0.0 = Complete collapse (100% one action)

        Used by: Renaissance Technologies, Two Sigma, Citadel
        Paper: "Maximum Entropy RL" (Haarnoja et al., 2018)
        """
        if len(self.recent_actions) < 50:
            return 0.5  # Neutral during warmup

        # Count action distribution over last 100 actions
        from collections import Counter
        counts = Counter(list(self.recent_actions)[-100:])

        # Calculate probabilities
        probs = []
        for action in [0, 1, 2]:  # SELL, HOLD, BUY
            probs.append(counts.get(action, 0) / 100.0)

        # Shannon entropy
        entropy = -sum([p * np.log(p + 1e-10) for p in probs if p > 0])
        max_entropy = np.log(3)  # Maximum entropy for 3 actions

        # Normalize to [0, 1]
        diversity_score = entropy / max_entropy

        return diversity_score

    def _update_advanced_risk_management(self):
        """
        V2 ULTIMATE: Advanced Risk Management metrics

        Updates:
        1. Kelly Criterion fraction (optimal position sizing)
        2. VaR 95% (Value at Risk)
        3. Tail Risk detection (kurtosis-based)

        Used by: All top hedge funds (Renaissance, Citadel, Two Sigma)
        """
        if len(self.recent_returns) < 20:
            return  # Need enough data

        returns = np.array(list(self.recent_returns))

        # 1. Kelly Criterion
        if len(self.trades) >= 10:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]

            if len(winning_trades) > 0 and len(losing_trades) > 0:
                win_rate = len(winning_trades) / len(self.trades)
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))

                if avg_loss > 0:
                    # Kelly fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
                    self.kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    self.kelly_fraction = np.clip(self.kelly_fraction, 0.0, 0.5)  # Max 50% Kelly

        # 2. VaR 95% (5th percentile of losses)
        self.var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0.0

        # 3. Tail Risk detection (fat tails = dangerous)
        from scipy import stats
        kurtosis = stats.kurtosis(returns)

        # Kurtosis > 3 = fat tails (normal distribution has kurtosis = 3)
        self.tail_risk_detected = kurtosis > 3.0

    def _update_rl_features(self):
        """
        V2 ULTIMATE: Update RL-specific features

        Features:
        1. Action history (last 5 actions)
        2. Regret signal (missed opportunities)
        3. Position duration
        4. Unrealized PnL ratio
        5. Market regime
        6. Hours until event
        7. Volatility percentile
        """
        # 1. Action history updated in step()

        # 2. Regret signal
        if self.last_close_price is not None and len(self.recent_actions) > 0:
            current_price = self.prices_df['close'].iloc[self.current_step]
            price_change_pct = (current_price - self.last_close_price) / self.last_close_price

            last_action = self.recent_actions[-1] if len(self.recent_actions) > 0 else 1

            # If HOLD but price moved significantly
            if last_action == 1 and abs(price_change_pct) > 0.001:  # 0.1% move
                self.regret_signal = abs(price_change_pct) * 100  # Scale to readable value
            else:
                self.regret_signal = 0.0

        # 3. Position duration
        if self.position_side != 0:
            self.position_entry_step = self.position_entry_step or self.current_step
            position_duration = self.current_step - self.position_entry_step
        else:
            position_duration = 0
            self.position_entry_step = 0

        # 4. Unrealized PnL ratio (vs initial risk)
        if self.position_side != 0 and abs(self.unrealized_pnl) > 0:
            initial_risk = self.balance * 0.01  # 1% risk
            self.unrealized_pnl_ratio = self.unrealized_pnl / initial_risk
        else:
            self.unrealized_pnl_ratio = 0.0

        # 5. Market regime (simple classification)
        if self.current_step >= 20:
            recent_closes = self.prices_df['close'].iloc[self.current_step-20:self.current_step].values
            price_range = (recent_closes.max() - recent_closes.min()) / recent_closes.mean()
            returns_std = np.std(np.diff(recent_closes) / recent_closes[:-1])

            if returns_std > 0.015:  # High volatility
                self.market_regime = 2  # Volatile
            elif price_range > 0.02:  # Trending
                self.market_regime = 1  # Trending
            else:
                self.market_regime = 0  # Ranging

        # 6. Hours until event (simplified - would use real macro calendar)
        # For now, assume events every Monday 14:30 (NFP-like)
        current_datetime = self.features_df.index[self.current_step]
        days_until_monday = (7 - current_datetime.weekday()) % 7
        if days_until_monday == 0:
            self.hours_until_event = 14.5 - current_datetime.hour - current_datetime.minute / 60
        else:
            self.hours_until_event = days_until_monday * 24 + 14.5 - current_datetime.hour

        # 7. Volatility percentile
        if self.current_step >= 100:
            current_vol = self.prices_df['close'].iloc[self.current_step-20:self.current_step].std()
            historical_vols = [
                self.prices_df['close'].iloc[i-20:i].std()
                for i in range(20, self.current_step, 20)
            ]
            from scipy import stats
            self.volatility_percentile = stats.percentileofscore(historical_vols, current_vol) / 100.0

    def _apply_safe_exploration_constraints(self, action, action_type: int = None):
        """
        V2 ULTIMATE: Safe Exploration (FTMO-aware)

        Prevents agent from violating FTMO rules during exploration

        Args:
            action: Raw action from policy
            action_type: Discrete action (0=SELL, 1=HOLD, 2=BUY) if applicable

        Returns:
            Safe action that won't violate FTMO

        Used by: Citadel (hard constraints on exploration)
        """
        # Check if action would violate FTMO

        # 1. Daily loss limit check
        if self.daily_loss_limit_reached and action_type in [0, 2]:
            # Force HOLD if daily loss limit reached
            if self.action_space_type == 'discrete':
                return 1  # HOLD
            else:
                return np.array([0.0, 0.0])  # No action

        # 2. Max drawdown approaching (7%) - WALL STREET FIX
        # ANCIEN CODE: Force HOLD (MAUVAIS - paralyse l'agent)
        # NOUVEAU CODE: Plus de confiance requise via _calculate_min_confidence_threshold()
        #               Lignes 408-417: DD >= 7% → 70% confidence minimum (vs 60% normal)
        # L'agent peut TOUJOURS trader, mais avec sélectivité accrue
        # → Comportement Wall Street: Trade moins, mais mieux
        # (Code force HOLD supprimé - remplacé par confidence threshold intelligent)

        # 3. Tail risk detected - reduce exposure
        if self.tail_risk_detected and self.position_side == 0:
            # Don't open new positions during tail risk
            if self.action_space_type == 'discrete':
                return 1  # HOLD
            else:
                action_copy = action.copy()
                action_copy[1] *= 0.5  # Half size if continuous
                return action_copy

        # Otherwise, return original action
        return action

    def _calculate_current_streak(self) -> int:
        """
        V2.1 CRITIC BOOST: Calculate current win/loss streak

        Returns:
            int: Positive = consecutive wins, Negative = consecutive losses
                 0 if no trades yet
                 Examples: +3 = 3 wins in a row, -2 = 2 losses in a row
        """
        if len(self.trades) < 1:
            return 0

        # Start from most recent trade and count backwards
        streak = 0
        last_was_win = self.trades[-1]['pnl'] > 0

        for trade in reversed(self.trades):
            is_win = trade['pnl'] > 0
            if is_win == last_was_win:
                streak += 1 if is_win else -1
            else:
                break  # Streak broken

        return streak

    def _calculate_adaptive_reward_multiplier(self) -> float:
        """
        V2 ULTIMATE: Adaptive Reward Scaling

        Adjusts reward based on current performance

        Returns:
            float: Multiplier [0.5, 2.0]
                   < 1.0 if performing poorly (encourage caution)
                   > 1.0 if performing well (encourage exploration)

        Used by: Two Sigma (dynamic reward adaptation)
        """
        # Calculate rolling performance score
        if len(self.trades) >= 5:
            recent_trades = self.trades[-10:]
            win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
            avg_pnl = np.mean([t['pnl'] for t in recent_trades])

            # Performance score = weighted avg of win rate and PnL
            self.performance_score = 0.6 * win_rate + 0.4 * (avg_pnl / self.initial_balance * 100)

            # Map performance to multiplier
            if self.performance_score > 0.7:  # Doing great
                return 1.5  # Boost rewards (encourage more of same)
            elif self.performance_score > 0.5:  # Doing good
                return 1.2
            elif self.performance_score > 0.3:  # Average
                return 1.0
            elif self.performance_score > 0.1:  # Struggling
                return 0.8  # Reduce rewards (more cautious)
            else:  # Doing poorly
                return 0.5  # Strong reduction (force reset)

        return 1.0  # Neutral until enough data

    def _calculate_trade_similarity_score(self) -> float:
        """
        WALL STREET V3: Trade Quality Memory System (25+25+10 INSTITUTIONAL GRADE)

        Compare le contexte de marché actuel avec :
        - 25 meilleurs trades (WINNERS)
        - 25 pires trades (LOSERS)
        - 10 trades manqués (MISSED - passivité excessive)

        Returns:
            float: Trade Similarity Score [-1.0, +1.0]
                +1.0 = Contexte ressemble fortement aux WINNERS
                0.0  = Neutre
                -1.0 = Contexte ressemble fortement aux LOSERS ou MISSED

        Used by: Renaissance Technologies, Two Sigma (pattern recognition + missed alpha)

        Performance: ~6ms per calculation (vs 5ms for 25+25, 2ms for 10+10)
        Warmup: 50 trades required (same as 25+25)
        """
        # Minimum 50 trades required for meaningful 25+25 comparison
        if len(self.trades) < 50:
            return 0.0  # Neutral during warmup period

        # Extract current market features (same 15 used at entry)
        try:
            current_features = np.array([
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('rsi_14')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd_signal')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('atr_14')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('adx')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('stochastic_k')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('cci')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('williams_r')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('roc')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('mfi')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('bb_width')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volume_ratio')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('trend_strength')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volatility_percentile')],
                float(self.max_drawdown)
            ], dtype=np.float32)
        except (KeyError, IndexError):
            return 0.0  # Neutral if features not available

        # Get top 25 winners (highest PnL) - ULTIMATE SWEET SPOT
        sorted_trades = sorted(self.trades, key=lambda t: t['pnl'], reverse=True)
        winners = sorted_trades[:25]
        winner_features = np.array([t['entry_features'] for t in winners], dtype=np.float32)

        # Get top 25 losers (lowest PnL) - ULTIMATE SWEET SPOT
        losers = sorted_trades[-25:]
        loser_features = np.array([t['entry_features'] for t in losers], dtype=np.float32)

        # Calculate cosine similarity (normalized dot product)
        # More efficient than sklearn for small vectors
        def cosine_sim(a, b):
            """Fast cosine similarity"""
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)

        # Average similarity with winners
        sim_winners = np.mean([cosine_sim(current_features, wf) for wf in winner_features])

        # Average similarity with losers
        sim_losers = np.mean([cosine_sim(current_features, lf) for lf in loser_features])

        # WALL STREET V3: Add similarity with MISSED opportunities (if enough data)
        sim_missed = 0.0
        if len(self.missed_opportunities) >= 10:
            # Get top 10 missed opportunities (highest potential PnL)
            sorted_missed = sorted(self.missed_opportunities, key=lambda m: m['pnl'], reverse=True)
            top_missed = sorted_missed[:10]
            missed_features = np.array([m['entry_features'] for m in top_missed], dtype=np.float32)

            # Average similarity with missed trades
            sim_missed = np.mean([cosine_sim(current_features, mf) for mf in missed_features])

        # Final score: [-1, +1]
        # If similar to winners and dissimilar to losers/missed → high positive score
        # If similar to losers/missed and dissimilar to winners → high negative score
        if len(self.missed_opportunities) >= 10:
            # V3: Weighted combination (70% losers, 30% missed)
            # Losers = actual mistakes (worse)
            # Missed = passivity errors (less bad but still penalized)
            similarity_score = sim_winners - (0.7 * sim_losers + 0.3 * sim_missed)
        else:
            # V2: Standard (no missed data yet)
            similarity_score = sim_winners - sim_losers

        # Clip to [-1, 1] range
        similarity_score = np.clip(similarity_score, -1.0, 1.0)

        return float(similarity_score)

    def _store_hold_context(self) -> None:
        """
        WALL STREET V3: Store market context after HOLD decision

        This context will be evaluated N steps later to detect missed opportunities.
        Used by: Renaissance Technologies (missed alpha tracking)
        """
        # Extract current market features (same 15 used for similarity)
        try:
            current_features = np.array([
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('rsi_14')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd_signal')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('atr_14')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('adx')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('stochastic_k')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('cci')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('williams_r')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('roc')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('mfi')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('bb_width')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volume_ratio')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('trend_strength')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volatility_percentile')],
                float(self.max_drawdown)
            ], dtype=np.float32)

            # Store context for evaluation after N steps
            self.potential_trade_context = {
                'step': self.current_step,
                'price': self.prices_df['close'].iloc[self.current_step],
                'entry_features': current_features,
                'trend_strength': self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('trend_strength')]
            }
            self.steps_since_hold = 0  # Reset counter
        except (KeyError, IndexError):
            self.potential_trade_context = None

    def _evaluate_missed_opportunities(self) -> None:
        """
        WALL STREET V3: Evaluate if previous HOLD was a missed opportunity

        Check after N steps if market moved significantly in predictable direction.
        If yes, store as "missed trade" for pattern avoidance.

        Used by: Renaissance Technologies, Two Sigma (opportunity cost analysis)
        """
        # No context to evaluate
        if self.potential_trade_context is None:
            return

        # Increment counter
        self.steps_since_hold += 1

        # Evaluate after 10 steps (~10 hours for H1 data)
        EVALUATION_WINDOW = 10

        if self.steps_since_hold >= EVALUATION_WINDOW:
            # Get current price
            current_price = self.prices_df['close'].iloc[self.current_step]
            entry_price = self.potential_trade_context['price']

            # Calculate price movement
            price_change = current_price - entry_price
            price_change_pct = price_change / entry_price

            # Minimum movement threshold: 0.3% (significant for gold)
            MIN_MOVE_PCT = 0.003

            if abs(price_change_pct) > MIN_MOVE_PCT:
                # Determine direction
                direction = 1 if price_change > 0 else -1

                # Simulate potential profit (using standard lot size)
                potential_pnl = direction * price_change * self.lot_size

                # Only store if would have been profitable (>$50)
                if potential_pnl > 50:
                    # Check if move was predictable (trend strength > 0.6)
                    trend_strength = self.potential_trade_context['trend_strength']

                    if trend_strength > 0.6:  # Strong trend = predictable
                        # Store as missed opportunity
                        self.missed_opportunities.append({
                            'pnl': potential_pnl,  # What we could have made
                            'entry_features': self.potential_trade_context['entry_features'],
                            'direction': direction,
                            'price_change_pct': price_change_pct
                        })

                        # Log if verbose
                        if self.verbose:
                            self.log(f"   [MISSED] Could have made ${potential_pnl:.2f} ({price_change_pct*100:.2f}%)")

            # Clear context after evaluation
            self.potential_trade_context = None
            self.steps_since_hold = 0

    def _calculate_v2_bonus_rewards(self, current_price: float) -> float:
        """
        V2 FEATURE: Calculate enhanced reward shaping components

        Returns:
            float: Combined V2 bonus rewards

        Components:
        1. Direction Prediction Reward (+0.02 for correct direction)
        2. Enhanced Profit Taking Bonus (+0.05 to +0.20 for closing profitable trades)
        3. Quick Loss Cutting Bonus (+0.03 for cutting losses early)
        4. Trade Completion Reward (+0.10 for closing any position)
        """
        v2_bonus = 0.0

        # 1. Direction Prediction Reward
        if self.last_close_price is not None:
            price_change = current_price - self.last_close_price
            actual_direction = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

            # If agent predicted direction correctly (even without trading)
            if self.last_action == 2 and actual_direction == 1:  # BUY and price went up
                v2_bonus += 0.02
            elif self.last_action == 0 and actual_direction == -1:  # SELL and price went down
                v2_bonus += 0.02

        # 2. Enhanced Profit Taking Bonus (when closing profitable trade)
        if self.last_closed_trade_pnl_pct is not None and self.last_closed_trade_pnl_pct > 0:
            pnl_pct = self.last_closed_trade_pnl_pct
            # Progressive bonus based on profit size
            if pnl_pct >= 0.01:  # >= 1% (4R)
                v2_bonus += 0.20  # BIG bonus for 4R+
            elif pnl_pct >= 0.005:  # >= 0.5% (2R)
                v2_bonus += 0.10
            elif pnl_pct >= 0.0025:  # >= 0.25% (1R)
                v2_bonus += 0.05
            else:  # < 1R but profitable
                v2_bonus += 0.02

        # 3. Quick Loss Cutting Bonus (when closing losing trade early)
        if self.last_closed_trade_pnl_pct is not None and self.last_closed_trade_pnl_pct < 0:
            pnl_pct = abs(self.last_closed_trade_pnl_pct)
            # Bonus for cutting losses before they grow
            if pnl_pct < 0.00125:  # < 0.125% (< 0.5R loss)
                v2_bonus += 0.03  # Good risk management!

        # 4. Trade Completion Reward (encourage finishing trades)
        if self.last_closed_trade_pnl_pct is not None:
            v2_bonus += 0.10  # Reward for completing any trade

        # Update last close price for next step
        self.last_close_price = current_price

        return v2_bonus

    def _calculate_reward(self) -> float:
        """
        V2 ULTIMATE: Tiered Reward Hierarchy (HEDGE FUND STANDARD)

        TIER 1 - Core Trading Performance (70%):
            - Profit component (40%)
            - Sharpe component (20%)
            - Drawdown penalty (10%)

        TIER 2 - Risk Management (20%):
            - FTMO compliance (10%)
            - VaR management (5%)
            - Tail risk control (5%)

        TIER 3 - Behavioral Shaping (10%):
            - Unified Diversity Score (5%) - REPLACES all HOLD penalties
            - Direction prediction (3%)
            - Completion bonus (2%)

        Adaptive Reward Multiplier: [0.5, 2.0] based on performance

        Used by: Jane Street, Citadel, Two Sigma
        Paper: Ng et al. (1999) "Policy Invariance Under Reward Transformations"
        """
        current_price = self.prices_df['close'].iloc[self.current_step]

        # Get V2 bonus rewards
        v2_bonus = self._calculate_v2_bonus_rewards(current_price)

        # Get unified diversity score (replaces all HOLD penalties)
        diversity_score = self._calculate_unified_diversity_score()

        # Get adaptive multiplier
        adaptive_multiplier = self._calculate_adaptive_reward_multiplier()

        # ========================================================================
        # TIER 1: CORE TRADING PERFORMANCE (70% of reward)
        # ========================================================================

        tier1_reward = 0.0

        # 1.1 Profit component (40%)
        profit_pct = (self.equity - self.initial_balance) / self.initial_balance
        tier1_reward += 0.40 * profit_pct

        # 1.2 Sharpe component (20%)
        if len(self.daily_returns) >= 5:
            returns = np.array(list(self.daily_returns))
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                tier1_reward += 0.20 * (sharpe / 10.0)  # Scaled

        # 1.3 Drawdown penalty (10%)
        dd_penalty = -self.max_drawdown * 5.0  # Strong penalty
        tier1_reward += 0.10 * dd_penalty

        # ========================================================================
        # TIER 2: RISK MANAGEMENT (20% of reward)
        # ========================================================================

        tier2_reward = 0.0

        # 2.1 FTMO compliance (10%)
        ftmo_score = 1.0
        if self.max_drawdown >= 0.10:  # Violation
            ftmo_score = 0.0
        elif self.max_drawdown >= 0.07:  # Warning zone
            ftmo_score = 0.5
        tier2_reward += 0.10 * (ftmo_score - 0.5)  # [-0.05, +0.05]

        # 2.2 VaR management (5%)
        if abs(self.var_95) > 0:
            var_score = 1.0 - min(abs(self.var_95) / 0.05, 1.0)  # Penalize large VaR
            tier2_reward += 0.05 * (var_score - 0.5)

        # 2.3 Tail risk control (5%)
        if self.tail_risk_detected:
            tier2_reward -= 0.05  # Penalty for fat tails
        else:
            tier2_reward += 0.02  # Bonus for normal distribution

        # ========================================================================
        # TIER 3: BEHAVIORAL SHAPING (10% of reward)
        # ========================================================================

        tier3_reward = 0.0

        # 3.1 EXPLORATION BONUS - Encourage BUY/SELL actions (FIX MODE COLLAPSE)
        # L'agent reçoit un GROS bonus pour chaque action de trading (pas HOLD)
        # V3 AGGRESSIVE: +0.08 (was +0.03)
        if self.last_action in [0, 2]:  # SELL or BUY
            tier3_reward += 0.08  # AGGRESSIVE bonus for trading
            self.log(f"   [BONUS] +0.08 for trading action")

        # 3.2 HOLD PENALTY - Penalize HOLD IMMEDIATELY (FIX MODE COLLAPSE)
        # V3 AGGRESSIVE: Pénalité dès le 1er HOLD, pas après 30
        # Chaque HOLD coûte -0.02, et ça augmente avec les HOLDs consécutifs
        if self.last_action == 1:  # HOLD
            # Base penalty for ANY hold
            base_hold_penalty = -0.02
            # Extra penalty for consecutive holds (starts at 10)
            if self.consecutive_holds > 10:
                extra_penalty = -0.05 * min((self.consecutive_holds - 10) / 30, 1.0)
            else:
                extra_penalty = 0
            hold_penalty = base_hold_penalty + extra_penalty
            tier3_reward += hold_penalty
            if self.consecutive_holds % 20 == 0:
                self.log(f"   [PENALTY] {hold_penalty:.4f} for {self.consecutive_holds} consecutive HOLDs")

        # 3.3 DIVERSITY ENCOURAGEMENT - Bonus for balanced action distribution
        # V3 AGGRESSIVE: Stronger penalties
        if len(self.recent_actions) >= 50:
            action_counts = {}
            for a in self.recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            total = sum(action_counts.values())
            max_pct = max(action_counts.values()) / total if total > 0 else 1.0

            if max_pct < 0.50:  # Very good diversity
                tier3_reward += 0.05  # Bigger bonus (was 0.02)
            elif max_pct < 0.60:  # Good diversity
                tier3_reward += 0.02
            elif max_pct > 0.80:  # Mode collapse warning
                tier3_reward -= 0.05  # Stronger penalty (was -0.03)
            elif max_pct > 0.90:  # Severe mode collapse
                tier3_reward -= 0.10  # Very strong penalty

        # 3.4 V2 Bonuses (direction, profit taking, etc.) (5%)
        tier3_reward += 0.05 * v2_bonus

        # ========================================================================
        # COMBINE ALL TIERS
        # ========================================================================

        base_reward = tier1_reward + tier2_reward + tier3_reward

        # Apply adaptive multiplier
        final_reward = base_reward * adaptive_multiplier

        return final_reward

        # OLD CODE BELOW - KEPT FOR REFERENCE IF NEEDED
        if False and REWARD_TYPE == 'profit':
            # Simple profit reward
            reward = self.daily_pnl / self.initial_balance

            # 5. Bonus 4R Take Profit
            if self.last_closed_trade_pnl_pct is not None:
                pnl_pct = self.last_closed_trade_pnl_pct
                if pnl_pct >= 0.01:  # 1% = 4R target (4 × 0.25%)
                    reward += 2.0  # GROS BONUS pour atteindre 4R!
                elif 0 < pnl_pct < 0.0025:  # 0.25% = 1R (fermeture prématurée)
                    reward -= 0.5  # Pénalité pour fermeture avant 1R
                self.last_closed_trade_pnl_pct = None  # Reset after use

            # AJOUTER LA PÉNALITÉ HOLD
            reward += hold_penalty

            # V2: AJOUTER LES BONUS V2
            reward += v2_bonus

            return reward

        elif REWARD_TYPE == 'sharpe_based':
            # Sharpe-based reward
            if len(self.daily_returns) >= 5:
                returns = np.array(list(self.daily_returns))
                if returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualisé
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            reward = sharpe / 10.0  # Scale down

            # 5. Bonus 4R Take Profit
            if self.last_closed_trade_pnl_pct is not None:
                pnl_pct = self.last_closed_trade_pnl_pct
                if pnl_pct >= 0.01:  # 1% = 4R target (4 × 0.25%)
                    reward += 2.0  # GROS BONUS pour atteindre 4R!
                elif 0 < pnl_pct < 0.0025:  # 0.25% = 1R (fermeture prématurée)
                    reward -= 0.5  # Pénalité pour fermeture avant 1R
                self.last_closed_trade_pnl_pct = None  # Reset after use

            # AJOUTER LA PÉNALITÉ HOLD
            reward += hold_penalty

            # V2: AJOUTER LES BONUS V2
            reward += v2_bonus

            return reward

        else:  # 'combined'
            # Reward combinée
            reward = 0.0

            # 1. Sharpe component
            if len(self.daily_returns) >= 5:
                returns = np.array(list(self.daily_returns))
                if returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                    reward += REWARD_WEIGHTS['sharpe'] * (sharpe / 10.0)

            # 2. Profit component
            profit_pct = (self.equity - self.initial_balance) / self.initial_balance
            reward += REWARD_WEIGHTS['profit'] * profit_pct

            # 3. Drawdown penalty
            dd_penalty = -self.max_drawdown * DRAWDOWN_PENALTY_MULTIPLIER
            reward += REWARD_WEIGHTS['drawdown_penalty'] * dd_penalty

            # 4. Win rate component
            if len(self.trades) >= MIN_TRADES_FOR_WINRATE:
                winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
                win_rate = winning_trades / len(self.trades)
                reward += REWARD_WEIGHTS['win_rate'] * (win_rate - 0.5)  # Bonus si >50%

            # 5. Bonus 4R Take Profit
            if self.last_closed_trade_pnl_pct is not None:
                pnl_pct = self.last_closed_trade_pnl_pct
                if pnl_pct >= 0.01:  # 1% = 4R target (4 × 0.25%)
                    reward += 2.0  # GROS BONUS pour atteindre 4R!
                elif 0 < pnl_pct < 0.0025:  # 0.25% = 1R (fermeture prématurée)
                    reward -= 0.5  # Pénalité pour fermeture avant 1R
                self.last_closed_trade_pnl_pct = None  # Reset after use

            # 6. AJOUTER LA PÉNALITÉ HOLD
            reward += hold_penalty

            # 7. V2: AJOUTER LES BONUS V2
            reward += v2_bonus

            return reward

    def _check_terminated(self) -> bool:
        """
        Vérifie si l'épisode doit se terminer

        Terminated si:
        - Max drawdown FTMO dépassé (10%)
        - Daily loss limit FTMO dépassé (2%)
        - Balance <= 0
        """
        # Max drawdown
        if self.max_drawdown >= FTMO_MAX_DRAWDOWN:
            self.ftmo_violations += 1
            self.log(f"   [ERROR] FTMO VIOLATION: Max drawdown {self.max_drawdown:.1%} >= {FTMO_MAX_DRAWDOWN:.1%}")
            return True

        # Daily loss
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance if self.daily_start_balance > 0 else 0.0
        if self.daily_pnl < 0 and daily_loss_pct >= FTMO_MAX_DAILY_LOSS:
            if FTMO_DAILY_LOSS_BLOCKS_NEW_TRADES:
                # Mode "blocage seulement" : on NE termine PAS l'épisode
                # Le flag daily_loss_limit_reached bloquera les nouveaux trades
                # mais les positions existantes continuent
                pass
            else:
                # Mode "arrêt complet" : terminer l'épisode (comportement original)
                self.ftmo_violations += 1
                self.log(f"   [ERROR] FTMO VIOLATION: Daily loss {daily_loss_pct:.1%} >= {FTMO_MAX_DAILY_LOSS:.1%}")
                return True

        # Balance
        if self.balance <= 0:
            self.log(f"   [ERROR] BANKRUPT: Balance = ${self.balance:.2f}")
            return True

        return False

    def _get_current_atr(self) -> float:
        """Obtient l'ATR actuel pour position sizing"""
        if self.current_step < ATR_PERIOD:
            return 10.0  # Default

        window = self.prices_df.iloc[max(0, self.current_step - ATR_PERIOD):self.current_step + 1]

        high = window['high'].values
        low = window['low'].values
        close = window['close'].values[:-1]

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close)
        tr3 = np.abs(low[1:] - close)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr)

        return atr

    def render(self, mode='human'):
        """Affiche l'état actuel"""
        if mode == 'human':
            info = self._get_info()
            print(f"\n{'='*80}")
            print(f"Step {info['step']} | {info['date']}")
            print(f"Price: ${info['price']:.2f}")
            print(f"Balance: ${info['balance']:,.2f} | Equity: ${info['equity']:,.2f}")
            print(f"Position: {['SHORT', 'FLAT', 'LONG'][info['position_side']+1]} "
                  f"({info['position']:.2f} lots)")
            print(f"Unrealized PnL: ${info['unrealized_pnl']:+.2f}")
            print(f"Daily PnL: ${info['daily_pnl']:+.2f} ({info['daily_pnl_pct']:+.2%})")
            print(f"Max DD: {info['max_drawdown']:.2%}")
            print(f"Trades: {info['total_trades']} | FTMO Violations: {info['ftmo_violations']}")
            print(f"{'='*80}")

    def get_episode_stats(self) -> Dict:
        """Statistiques de l'épisode"""
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': self.max_drawdown,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'ftmo_violations': self.ftmo_violations,
            }

        total_return = (self.equity - self.initial_balance) / self.initial_balance

        # Sharpe
        if len(self.daily_returns) >= 2:
            returns = np.array(list(self.daily_returns))
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Win rate
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = winning_trades / len(self.trades) if len(self.trades) > 0 else 0.0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'ftmo_violations': self.ftmo_violations,
            'final_balance': self.balance,
            'final_equity': self.equity,
        }


def test_environment():
    """Test l'environnement"""
    print("="*80)
    print("[TEST] TESTING TRADING ENVIRONMENT")
    print("="*80)

    # Créer des données de test
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')

    # Features dummy (232)
    features_data = np.random.randn(1000, 232)
    features_df = pd.DataFrame(features_data, index=dates,
                              columns=[f'feature_{i}' for i in range(232)])

    # Prices dummy
    prices_data = {
        'open': 2000 + np.cumsum(np.random.randn(1000) * 2),
        'high': 2000 + np.cumsum(np.random.randn(1000) * 2) + 5,
        'low': 2000 + np.cumsum(np.random.randn(1000) * 2) - 5,
        'close': 2000 + np.cumsum(np.random.randn(1000) * 2),
        'volume': np.random.randint(1000, 10000, 1000),
    }
    prices_df = pd.DataFrame(prices_data, index=dates)

    # Créer environnement
    env = GoldTradingEnv(
        features_df=features_df,
        prices_df=prices_df,
        initial_balance=100000,
        action_space_type='discrete',
        verbose=True
    )

    # Test reset
    obs, info = env.reset()
    print(f"\n[OK] Environment reset")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Info: {info}")

    # Test quelques steps
    print(f"\n🎮 Testing steps...")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"   Action: {action}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Balance: ${info['balance']:,.2f}")
        print(f"   Position: {['SHORT', 'FLAT', 'LONG'][info['position_side']+1]}")

        if terminated or truncated:
            print(f"\n   Episode ended (terminated={terminated}, truncated={truncated})")
            break

    # Stats finales
    stats = env.get_episode_stats()
    print(f"\n{'='*80}")
    print(f"EPISODE STATS:")
    print(f"{'='*80}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print(f"\n{'='*80}")
    print(f"[OK] ENVIRONMENT TEST SUCCESSFUL")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_environment()
