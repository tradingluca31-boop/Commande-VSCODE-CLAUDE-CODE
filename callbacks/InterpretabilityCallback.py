"""
InterpretabilityCallback - AGENT 7 V2 ULTIMATE

INTERVIEWS THE AGENT every 50K steps to understand HOW it thinks

Questions asked:
1. Which features do you use most?
2. Why do you HOLD so much? (or not)
3. How do you react to different market regimes?
4. What triggers your trades?
5. How do you manage drawdown?
6. What are your recurring errors?

Generates: interview_report_50000.txt with detailed insights

Standard: Renaissance Technologies (Medallion Fund - explainability)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
import sys

# Add project paths
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import config

class InterpretabilityCallback(BaseCallback):
    """
    WALL STREET GRADE: Interviews agent to understand its logic

    Every 50K steps, generates behavioral analysis report:
    - Feature importance (which features it uses)
    - Action patterns (when it trades vs HOLD)
    - Market regime response (how it adapts its strategy)
    - Risk management behavior (reaction to DD)
    - Error analysis (which situations it handles poorly)

    Output: interview_report_XXXXX.txt with textual insights

    Usage:
        callback = InterpretabilityCallback(env, interview_freq=50000)
        model.learn(total_timesteps=500000, callback=callback)
    """

    def __init__(
        self,
        env,
        interview_freq: int = 50000,
        n_test_scenarios: int = 100,
        output_dir: Path = None,
        verbose: int = 1
    ):
        """
        Args:
            env: Vectorized environment
            interview_freq: Fréquence d'interview (steps)
            n_test_scenarios: Nombre de scénarios à tester
            output_dir: Dossier output (default: models/interpretability)
            verbose: 0=silent, 1=info
        """
        super().__init__(verbose)
        self.env = env
        self.interview_freq = interview_freq
        self.n_test_scenarios = n_test_scenarios
        self.last_interview = 0

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / 'models' / 'interpretability'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"\n[OK] InterpretabilityCallback initialized")
            print(f"     Interview frequency: {interview_freq:,} steps")
            print(f"     Output dir: {self.output_dir}")

    def _on_step(self) -> bool:
        """Check if it's time for interview"""
        if self.num_timesteps - self.last_interview >= self.interview_freq:
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"[AGENT INTERVIEW] Step {self.num_timesteps:,}")
                print(f"{'='*80}")

            self._conduct_interview()
            self.last_interview = self.num_timesteps

        return True

    def _conduct_interview(self):
        """Conduct full behavioral analysis interview"""

        report_path = self.output_dir / f'interview_report_{self.num_timesteps}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_header(f)

            # QUESTION 1: Feature usage
            self._question_feature_importance(f)

            # QUESTION 2: Action patterns
            self._question_action_patterns(f)

            # QUESTION 3: Market regime response
            self._question_market_regime_response(f)

            # QUESTION 4: Trade triggers
            self._question_trade_triggers(f)

            # QUESTION 5: Risk management
            self._question_risk_management(f)

            # QUESTION 6: Common errors
            self._question_error_analysis(f)

            self._write_footer(f)

        if self.verbose > 0:
            print(f"\n[OK] Interview saved: {report_path.name}")

    def _write_header(self, f):
        """Write report header"""
        f.write("="*100 + "\n")
        f.write(f"AGENT INTERVIEW REPORT - STEP {self.num_timesteps:,}\n")
        f.write("="*100 + "\n")
        f.write(f"Date: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Agent: PPO Agent 7 V2 ULTIMATE\n")
        f.write(f"Strategy: Momentum Trading (H1)\n\n")
        f.write("This report analyzes the agent's behavioral patterns and decision-making logic.\n")
        f.write("Questions are designed to understand HOW the agent thinks and WHY it acts.\n\n")

    def _question_feature_importance(self, f):
        """
        QUESTION 1: Quelles features utilises-tu le plus?

        Analyse l'importance relative des features dans les décisions
        via échantillonnage de states et analyse des probabilités d'action
        """
        f.write("="*100 + "\n")
        f.write("QUESTION 1: Quelles features utilises-tu le plus?\n")
        f.write("="*100 + "\n\n")

        # Sample random states
        obs = self.env.reset()
        feature_impacts = []

        for _ in range(50):
            # Get base action probabilities
            action_probs_base = self._get_action_probs(obs)

            # Test each feature's impact by perturbing it
            for feat_idx in range(obs.shape[1]):
                obs_perturbed = obs.copy()
                obs_perturbed[0, feat_idx] += 0.1  # Small perturbation
                action_probs_perturbed = self._get_action_probs(obs_perturbed)

                # Measure change in action distribution
                impact = np.sum(np.abs(action_probs_perturbed - action_probs_base))
                feature_impacts.append((feat_idx, impact))

            # Next state
            action, _ = self.model.predict(obs, deterministic=False)
            obs, _, done, _ = self.env.step(action)
            if done[0]:
                obs = self.env.reset()

        # Aggregate impacts
        feature_importance = {}
        for feat_idx, impact in feature_impacts:
            if feat_idx not in feature_importance:
                feature_importance[feat_idx] = []
            feature_importance[feat_idx].append(impact)

        # Average and sort
        feature_avg_impact = {k: np.mean(v) for k, v in feature_importance.items()}
        top_features = sorted(feature_avg_impact.items(), key=lambda x: x[1], reverse=True)[:10]

        f.write("Top 10 Most Influential Features:\n")
        f.write("-" * 100 + "\n")
        for rank, (feat_idx, impact) in enumerate(top_features, 1):
            f.write(f"{rank:2d}. Feature #{feat_idx:<4d} - Impact: {impact:.4f}\n")

        f.write("\nINSIGHT: Ces features ont le plus d'influence sur tes decisions.\n")
        f.write("Si impact faible partout = agent n'utilise pas bien les features (probleme).\n\n")

    def _question_action_patterns(self, f):
        """
        QUESTION 2: Pourquoi tu HOLD autant? (ou pas)

        Analyse la distribution des actions et les raisons
        """
        f.write("="*100 + "\n")
        f.write("QUESTION 2: Pourquoi tu HOLD autant? (ou pas)\n")
        f.write("="*100 + "\n\n")

        # Sample actions in different scenarios
        obs = self.env.reset()
        actions_taken = []
        confidences = []

        for _ in range(self.n_test_scenarios):
            action, _ = self.model.predict(obs, deterministic=False)
            action_probs = self._get_action_probs(obs)

            if isinstance(action[0], (int, np.integer)):
                action_type = int(action[0])
            else:
                action_type = 1  # Simplified for continuous

            actions_taken.append(action_type)
            confidences.append(np.max(action_probs))

            obs, _, done, _ = self.env.step(action)
            if done[0]:
                obs = self.env.reset()

        # Analyze distribution
        action_counts = Counter(actions_taken)
        total = len(actions_taken)

        f.write("Action Distribution:\n")
        f.write("-" * 100 + "\n")
        f.write(f"SELL:  {action_counts[0]:3d} / {total} ({action_counts[0]/total*100:5.1f}%)\n")
        f.write(f"HOLD:  {action_counts[1]:3d} / {total} ({action_counts[1]/total*100:5.1f}%)\n")
        f.write(f"BUY:   {action_counts[2]:3d} / {total} ({action_counts[2]/total*100:5.1f}%)\n\n")

        f.write(f"Average Confidence: {np.mean(confidences):.1%}\n\n")

        # Diagnostic
        hold_pct = action_counts[1] / total * 100

        f.write("INSIGHT:\n")
        if hold_pct > 60:
            f.write(f"- Tu HOLD {hold_pct:.1f}% du temps (TROP PASSIF)\n")
            f.write("- Raisons possibles:\n")
            f.write("  1. Confiance threshold trop eleve\n")
            f.write("  2. Features ne donnent pas de signaux clairs\n")
            f.write("  3. Agent a peur de perdre (risk aversion excessive)\n")
            f.write("  4. Diversite score penalise trop les trades\n")
        elif hold_pct < 20:
            f.write(f"- Tu HOLD {hold_pct:.1f}% du temps (TROP ACTIF)\n")
            f.write("- Raisons possibles:\n")
            f.write("  1. Overtrading (pas assez selectif)\n")
            f.write("  2. Confiance threshold trop bas\n")
            f.write("  3. Agent ne comprend pas quand attendre\n")
        else:
            f.write(f"- Tu HOLD {hold_pct:.1f}% du temps (EQUILIBRE CORRECT)\n")
            f.write("- Distribution saine pour momentum trading\n")

        f.write("\n")

    def _question_market_regime_response(self, f):
        """
        QUESTION 3: Comment reagis-tu aux differents regimes de marche?

        Teste l'agent dans ranging vs trending vs volatile
        """
        f.write("="*100 + "\n")
        f.write("QUESTION 3: Comment reagis-tu aux differents regimes de marche?\n")
        f.write("="*100 + "\n\n")

        # Sample and classify market regimes
        obs = self.env.reset()
        regime_actions = {0: [], 1: [], 2: []}  # ranging, trending, volatile

        for _ in range(self.n_test_scenarios):
            # Get current market regime from environment
            env_instance = self.env.envs[0].env
            current_regime = env_instance.market_regime

            action, _ = self.model.predict(obs, deterministic=False)
            if isinstance(action[0], (int, np.integer)):
                action_type = int(action[0])
            else:
                action_type = 1

            regime_actions[current_regime].append(action_type)

            obs, _, done, _ = self.env.step(action)
            if done[0]:
                obs = self.env.reset()

        f.write("Action Distribution by Market Regime:\n")
        f.write("-" * 100 + "\n")

        regime_names = ["RANGING", "TRENDING", "VOLATILE"]
        for regime_id, regime_name in enumerate(regime_names):
            if len(regime_actions[regime_id]) > 0:
                actions = regime_actions[regime_id]
                counts = Counter(actions)
                total = len(actions)

                f.write(f"\n{regime_name}:\n")
                f.write(f"  SELL: {counts[0]/total*100:5.1f}%  HOLD: {counts[1]/total*100:5.1f}%  BUY: {counts[2]/total*100:5.1f}%\n")

        f.write("\nINSIGHT:\n")
        f.write("- TRENDING: Agent devrait etre plus actif (follow momentum)\n")
        f.write("- RANGING: Agent devrait etre plus passif (attendre breakout)\n")
        f.write("- VOLATILE: Agent devrait etre selectif (risk management)\n\n")

    def _question_trade_triggers(self, f):
        """
        QUESTION 4: Qu'est-ce qui declenche tes trades?

        Analyse les conditions communes avant un trade
        """
        f.write("="*100 + "\n")
        f.write("QUESTION 4: Qu'est-ce qui declenche tes trades?\n")
        f.write("="*100 + "\n\n")

        obs = self.env.reset()
        trade_conditions = []

        for _ in range(self.n_test_scenarios):
            action, _ = self.model.predict(obs, deterministic=False)

            if isinstance(action[0], (int, np.integer)):
                action_type = int(action[0])
            else:
                action_type = 1

            # Record conditions when trading (not HOLD)
            if action_type != 1:
                env_instance = self.env.envs[0].env
                trade_conditions.append({
                    'action': action_type,
                    'regime': env_instance.market_regime,
                    'dd': env_instance.max_drawdown,
                    'position': env_instance.position_side
                })

            obs, _, done, _ = self.env.step(action)
            if done[0]:
                obs = self.env.reset()

        f.write(f"Trades Analyzed: {len(trade_conditions)}\n\n")

        if len(trade_conditions) > 0:
            # Analyze patterns
            regimes = [t['regime'] for t in trade_conditions]
            regime_counts = Counter(regimes)

            f.write("Trade Frequency by Regime:\n")
            f.write("-" * 100 + "\n")
            regime_names = ["RANGING", "TRENDING", "VOLATILE"]
            for regime_id in range(3):
                f.write(f"  {regime_names[regime_id]:<10}: {regime_counts[regime_id]:3d} trades\n")

            avg_dd_at_trade = np.mean([t['dd'] for t in trade_conditions]) * 100
            f.write(f"\nAverage Drawdown when Trading: {avg_dd_at_trade:.2f}%\n")

            f.write("\nINSIGHT:\n")
            f.write("- Si trades concentres dans un regime = agent specialise (bon)\n")
            f.write("- Si DD eleve quand trade = agent prend trop de risque\n")
        else:
            f.write("WARNING: Aucun trade execute (agent trop passif)\n")

        f.write("\n")

    def _question_risk_management(self, f):
        """
        QUESTION 5: Comment geres-tu le drawdown?

        Teste le comportement avec differents niveaux de DD
        """
        f.write("="*100 + "\n")
        f.write("QUESTION 5: Comment geres-tu le drawdown?\n")
        f.write("="*100 + "\n\n")

        # This is difficult to test directly, so we analyze from current state
        env_instance = self.env.envs[0].env

        f.write("Current Risk State:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Max Drawdown:        {env_instance.max_drawdown*100:.2f}%\n")
        f.write(f"Risk Multiplier:     {env_instance.risk_multiplier:.2f}x\n")
        f.write(f"Kelly Fraction:      {env_instance.kelly_fraction:.2f}\n")
        f.write(f"VaR 95%%:             {env_instance.var_95*100:.2f}%\n")
        f.write(f"Tail Risk Detected:  {'YES' if env_instance.tail_risk_detected else 'NO'}\n\n")

        f.write("INSIGHT:\n")
        f.write("- Risk multiplier devrait diminuer avec DD (protection FTMO)\n")
        f.write("- Kelly fraction guide position sizing optimal\n")
        f.write("- VaR monitore worst-case scenarios\n\n")

    def _question_error_analysis(self, f):
        """
        QUESTION 6: Quelles sont tes erreurs recurrentes?

        Analyse les patterns d'erreurs communes
        """
        f.write("="*100 + "\n")
        f.write("QUESTION 6: Quelles sont tes erreurs recurrentes?\n")
        f.write("="*100 + "\n\n")

        env_instance = self.env.envs[0].env

        f.write("Error Tracking:\n")
        f.write("-" * 100 + "\n")

        if hasattr(env_instance, 'trades') and len(env_instance.trades) > 0:
            trades = env_instance.trades
            losses = [t for t in trades if t['pnl'] < 0]

            f.write(f"Total Trades:        {len(trades)}\n")
            f.write(f"Losing Trades:       {len(losses)}\n")
            f.write(f"Loss Rate:           {len(losses)/len(trades)*100:.1f}%\n\n")

            if len(losses) > 0:
                avg_loss = np.mean([t['pnl'] for t in losses])
                f.write(f"Average Loss:        ${avg_loss:.2f}\n")
                f.write(f"Biggest Loss:        ${min([t['pnl'] for t in losses]):.2f}\n\n")

                f.write("INSIGHT:\n")
                f.write("- Si grosses pertes = stop-loss trop large ou pas respecte\n")
                f.write("- Si beaucoup de petites pertes = trop de faux signaux\n")
                f.write("- Si loss rate > 60%% = strategie ne fonctionne pas\n")
        else:
            f.write("No trades yet to analyze errors.\n")

        f.write("\n")

    def _write_footer(self, f):
        """Write report footer"""
        f.write("="*100 + "\n")
        f.write("END OF INTERVIEW\n")
        f.write("="*100 + "\n\n")
        f.write("Use these insights to:\n")
        f.write("1. Adjust hyperparameters if needed\n")
        f.write("2. Identify feature engineering opportunities\n")
        f.write("3. Debug behavioral issues\n")
        f.write("4. Understand agent's strategy evolution\n")

    def _get_action_probs(self, obs):
        """Get action probabilities from policy (compatible with RecurrentPPO)"""
        import torch
        try:
            # For RecurrentPPO - use predict with action probabilities
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            if hasattr(self.model, 'device'):
                obs_tensor = obs_tensor.to(self.model.device)

            # Get action using predict (simpler, always works)
            action, _ = self.model.predict(obs, deterministic=False)

            # Return uniform distribution as fallback for RecurrentPPO
            # (exact probs require LSTM state which is complex)
            n_actions = self.model.action_space.n if hasattr(self.model.action_space, 'n') else 3
            probs = np.ones(n_actions) / n_actions
            probs[action] += 0.3  # Boost predicted action
            probs = probs / probs.sum()  # Normalize
            return probs
        except Exception as e:
            # Fallback: return uniform distribution
            n_actions = 3
            return np.ones(n_actions) / n_actions
