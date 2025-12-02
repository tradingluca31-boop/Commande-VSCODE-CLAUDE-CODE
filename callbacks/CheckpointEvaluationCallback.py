"""
CheckpointEvaluationCallback - AGENT 7 V2 ULTIMATE

ENSURES data saving during training (like Agent 8)

Generates AUTOMATICALLY every 50K steps:
  - checkpoint_50000_stats.csv (aggregated metrics)
  - checkpoint_50000_trades.csv (detail of each trade)

Duration: ~10 seconds every 50K steps (overhead <0.1%)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import sys

# Add project paths
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import config

class CheckpointEvaluationCallback(BaseCallback):
    """
    WALL STREET GRADE: Evaluates model every 50K steps and saves CSV

    System identical to Agent 8 - PRODUCTION READY

    Saves:
    1. Stats CSV: Balance, Win Rate, Profit Factor, Actions, etc.
    2. Trades CSV: Detail of each trade (entry, exit, pnl, side, etc.)

    Usage:
        callback = CheckpointEvaluationCallback(env, eval_freq=50000)
        model.learn(total_timesteps=500000, callback=callback)
    """

    def __init__(
        self,
        env,
        eval_freq: int = 50000,
        n_eval_steps: int = 500,
        output_dir: Optional[Path] = None,
        verbose: int = 1
    ):
        """
        Args:
            env: Vectorized environment (DummyVecEnv)
            eval_freq: Fréquence d'évaluation (steps)
            n_eval_steps: Nombre de steps pour évaluation
            output_dir: Dossier de sortie (default: models/checkpoints_analysis)
            verbose: 0=silent, 1=info
        """
        super().__init__(verbose)
        self.env = env
        self.eval_freq = eval_freq
        self.n_eval_steps = n_eval_steps
        self.last_eval = 0

        # Output directory
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / 'models' / 'checkpoints_analysis'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"\n[OK] CheckpointEvaluationCallback initialized")
            print(f"     Eval frequency: {eval_freq:,} steps")
            print(f"     Output dir: {self.output_dir}")

    def _on_step(self) -> bool:
        """Called at each step - check if evaluation needed"""

        # Check if it's time to evaluate
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"[CHECKPOINT EVALUATION] Step {self.num_timesteps:,}")
                print(f"{'='*80}")

            t0 = dt.now()

            # Run evaluation
            self._evaluate_and_save()

            t1 = dt.now()
            duration = (t1 - t0).total_seconds()

            if self.verbose > 0:
                print(f"\n[OK] Evaluation completed in {duration:.1f}s")
                print(f"{'='*80}\n")

            self.last_eval = self.num_timesteps

        return True

    def _evaluate_and_save(self):
        """
        Évalue le modèle sur N steps et sauvegarde CSV

        CRITICAL: Utilise le flag 'trade_closed' de l'environnement
        pour capturer TOUS les trades pendant l'évaluation
        """
        # Reset environment
        obs = self.env.reset()

        # Tracking
        trades = []
        actions_taken = [0, 0, 0]  # SELL, HOLD, BUY

        # Initial state
        info = self.env.env_method('_get_info')[0]
        initial_balance = info['balance']

        # Run N evaluation steps
        for step in range(self.n_eval_steps):
            # Predict action (deterministic)
            action, _states = self.model.predict(obs, deterministic=True)

            # Execute
            obs, reward, done, info_list = self.env.step(action)
            info = info_list[0]  # Get info from first env (DummyVecEnv)

            # Track action type
            if isinstance(action[0], (int, np.integer)):
                action_type = int(action[0])
            else:
                # Continuous action space - classify
                if abs(action[0][0]) < 0.3 or action[0][1] < 0.1:
                    action_type = 1  # HOLD
                else:
                    action_type = 0 if action[0][0] < 0 else 2  # SELL or BUY

            actions_taken[action_type] += 1

            # Capture trade if closed
            if info.get('trade_closed', False):
                trade_info = info.get('last_trade', None)
                if trade_info is not None:
                    trade_record = {
                        'checkpoint': self.num_timesteps,
                        'step': step,
                        'entry_price': trade_info['entry_price'],
                        'exit_price': trade_info['exit_price'],
                        'side': trade_info['side'],
                        'size': trade_info['size'],
                        'pnl': trade_info['pnl'],
                        'pnl_pct': trade_info['pnl_pct'],
                        'balance_after': trade_info['balance_after'],
                        'win': trade_info['win']
                    }
                    trades.append(trade_record)

            # Reset if done
            if done[0]:
                obs = self.env.reset()

        # Final state
        final_info = self.env.env_method('_get_info')[0]
        final_balance = final_info['balance']

        # Calculate metrics
        total_trades = len(trades)
        win_rate = 0.0
        profit_factor = 0.0
        total_pnl = final_balance - initial_balance

        if total_trades > 0:
            wins = [t for t in trades if t['win'] == 1]
            losses = [t for t in trades if t['win'] == 0]

            win_rate = len(wins) / total_trades * 100

            if len(wins) > 0 and len(losses) > 0:
                total_wins = sum(t['pnl'] for t in wins)
                total_losses = abs(sum(t['pnl'] for t in losses))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Action distribution percentages
        action_sell_pct = actions_taken[0] / self.n_eval_steps * 100
        action_hold_pct = actions_taken[1] / self.n_eval_steps * 100
        action_buy_pct = actions_taken[2] / self.n_eval_steps * 100

        # Save STATS CSV
        stats_df = pd.DataFrame([{
            'checkpoint': self.num_timesteps,
            'timestamp': dt.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'action_sell_pct': action_sell_pct,
            'action_hold_pct': action_hold_pct,
            'action_buy_pct': action_buy_pct,
            'max_drawdown': final_info.get('max_drawdown', 0.0) * 100
        }])

        stats_csv = self.output_dir / f'checkpoint_{self.num_timesteps}_stats.csv'
        stats_df.to_csv(stats_csv, index=False)

        if self.verbose > 0:
            print(f"\n[STATS]")
            print(f"  Balance: ${initial_balance:,.0f} -> ${final_balance:,.0f} ({total_pnl:+.2f})")
            print(f"  Trades:  {total_trades}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Actions: SELL {action_sell_pct:.1f}% | HOLD {action_hold_pct:.1f}% | BUY {action_buy_pct:.1f}%")
            print(f"\n  Saved: {stats_csv.name}")

        # Save TRADES CSV (if any)
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_csv = self.output_dir / f'checkpoint_{self.num_timesteps}_trades.csv'
            trades_df.to_csv(trades_csv, index=False)

            if self.verbose > 0:
                print(f"  Saved: {trades_csv.name}")
        else:
            if self.verbose > 0:
                print(f"  [WARNING] No trades captured (agent too passive or confidence too low)")

    def _on_training_end(self):
        """Generate final ranking when training ends"""
        if self.verbose > 0:
            print(f"\n{'='*80}")
            print(f"[CHECKPOINT EVALUATION] Training finished")
            print(f"{'='*80}")

        # Generate ranking from all saved checkpoints
        self._generate_ranking()

    def _generate_ranking(self):
        """Generate RANKING.csv and RANKING.txt from all checkpoint stats"""

        # Find all stats CSV files
        stats_files = sorted(self.output_dir.glob('checkpoint_*_stats.csv'))

        if not stats_files:
            if self.verbose > 0:
                print(f"[WARNING] No stats files found for ranking")
            return

        # Load all stats
        all_stats = []
        for csv_file in stats_files:
            df = pd.read_csv(csv_file)
            all_stats.append(df.iloc[0])

        results_df = pd.DataFrame(all_stats)

        # Calculate composite score
        results_df['roi_pct'] = (results_df['final_balance'] - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100

        # Score /10
        results_df['score'] = (
            results_df['roi_pct'] / 2 * 0.30 +  # ROI
            (results_df['win_rate'] - 40) / 2 * 0.20 +  # Win Rate
            (results_df['profit_factor'] - 0.8) * 10 * 0.20 +  # PF
            (100 - results_df['max_drawdown']) / 10 * 0.10  # DD
        )
        results_df['score'] = results_df['score'].clip(0, 10)

        # Sort by score
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)

        # Save CSV
        ranking_csv = self.output_dir / 'RANKING.csv'
        results_df.to_csv(ranking_csv, index=False)

        # Save TXT
        ranking_txt = self.output_dir / 'RANKING.txt'
        with open(ranking_txt, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("RANKING - ALL CHECKPOINTS AGENT 7 V2 ULTIMATE\n")
            f.write("="*100 + "\n")
            f.write(f"Date: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Checkpoints: {len(results_df)}\n\n")

            # Header
            f.write(f"{'Rank':<6} {'Step':<10} {'Score':<8} {'ROI%':<8} {'WR%':<7} {'PF':<6} {'DD%':<7} {'Trades':<8}\n")
            f.write("="*100 + "\n")

            # Rows
            for idx, row in results_df.iterrows():
                rank = idx + 1
                if rank == 1:
                    medal = "1st"
                elif rank == 2:
                    medal = "2nd"
                elif rank == 3:
                    medal = "3rd"
                else:
                    medal = f"{rank}th"

                line = (f"{medal:<6} {int(row['checkpoint']):<10} {row['score']:>6.2f}/10 "
                       f"{row['roi_pct']:>6.2f}% {row['win_rate']:>5.1f}% "
                       f"{row['profit_factor']:>5.2f} {row['max_drawdown']:>5.1f}% {int(row['total_trades']):>6}\n")
                f.write(line)

            f.write("="*100 + "\n\n")

            # Best checkpoint
            best = results_df.iloc[0]
            f.write("BEST CHECKPOINT\n")
            f.write("="*100 + "\n")
            f.write(f"Checkpoint:    {int(best['checkpoint']):,} steps\n")
            f.write(f"Score:         {best['score']:.2f}/10\n\n")
            f.write(f"Balance:       ${best['final_balance']:,.0f}\n")
            f.write(f"ROI:           {best['roi_pct']:+.2f}%\n")
            f.write(f"Win Rate:      {best['win_rate']:.1f}%\n")
            f.write(f"Profit Factor: {best['profit_factor']:.2f}\n")
            f.write(f"Max Drawdown:  {best['max_drawdown']:.1f}%\n")
            f.write(f"Total Trades:  {int(best['total_trades'])}\n")

        if self.verbose > 0:
            print(f"\n[RANKING] Generated:")
            print(f"  - {ranking_csv.name}")
            print(f"  - {ranking_txt.name}")
            print(f"\nBEST CHECKPOINT: {int(best['checkpoint']):,} steps (Score: {best['score']:.2f}/10)")
