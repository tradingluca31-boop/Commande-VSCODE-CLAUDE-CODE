"""
COMPREHENSIVE CHECKPOINT ANALYSIS
Analyze ALL interviews + trades to find TRUE best model
"""

import pandas as pd
import os
from pathlib import Path
import re

# Paths
BASE_PATH = Path(__file__).parent.parent / "models"
INTERPRETABILITY_PATH = BASE_PATH / "interpretability"
CHECKPOINTS_ANALYSIS_PATH = BASE_PATH / "checkpoints_analysis"

def extract_interview_metrics(interview_path):
    """Extract key metrics from interview report."""
    with open(interview_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {}

    # Extract checkpoint number
    match = re.search(r'STEP (\d+)', content)
    metrics['checkpoint'] = int(match.group(1)) if match else 0

    # Extract action distribution
    sell_match = re.search(r'SELL:\s+(\d+)\s*/\s*100\s*\(\s*([\d.]+)%\)', content)
    hold_match = re.search(r'HOLD:\s+(\d+)\s*/\s*100\s*\(\s*([\d.]+)%\)', content)
    buy_match = re.search(r'BUY:\s+(\d+)\s*/\s*100\s*\(\s*([\d.]+)%\)', content)

    metrics['sell_pct'] = float(sell_match.group(2)) if sell_match else 0
    metrics['hold_pct'] = float(hold_match.group(2)) if hold_match else 0
    metrics['buy_pct'] = float(buy_match.group(2)) if buy_match else 0

    # Extract confidence
    conf_match = re.search(r'Average Confidence:\s*([\d.]+)%', content)
    metrics['avg_confidence'] = float(conf_match.group(1)) if conf_match else 0

    # Extract drawdown
    dd_match = re.search(r'Max Drawdown:\s*([\d.]+)%', content)
    metrics['max_dd'] = float(dd_match.group(1)) if dd_match else 0

    # Extract trade metrics
    total_trades_match = re.search(r'Total Trades:\s*(\d+)', content)
    losing_trades_match = re.search(r'Losing Trades:\s*(\d+)', content)
    loss_rate_match = re.search(r'Loss Rate:\s*([\d.]+)%', content)

    metrics['total_trades_interview'] = int(total_trades_match.group(1)) if total_trades_match else 0
    metrics['losing_trades'] = int(losing_trades_match.group(1)) if losing_trades_match else 0
    metrics['loss_rate'] = float(loss_rate_match.group(1)) if loss_rate_match else 0
    metrics['win_rate_interview'] = 100 - metrics['loss_rate']

    # Extract biggest loss
    biggest_loss_match = re.search(r'Biggest Loss:\s*\$(-?[\d.,]+)', content)
    if biggest_loss_match:
        metrics['biggest_loss'] = float(biggest_loss_match.group(1).replace(',', ''))
    else:
        metrics['biggest_loss'] = 0

    # Calculate diversity score (Shannon entropy)
    from math import log
    actions = [metrics['sell_pct'], metrics['hold_pct'], metrics['buy_pct']]
    actions_norm = [a/100 for a in actions if a > 0]
    diversity = -sum(p * log(p, 3) for p in actions_norm if p > 0)
    metrics['diversity_score'] = diversity

    return metrics

def analyze_trades_csv(trades_path):
    """Analyze trades from CSV."""
    if not os.path.exists(trades_path):
        return {}

    df = pd.read_csv(trades_path)

    if len(df) == 0:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': 0,
            'consecutive_losses_max': 0
        }

    # Basic stats
    metrics = {
        'num_trades': len(df),
        'win_rate': (df['pnl'] > 0).mean() * 100,
        'avg_profit': df[df['pnl'] > 0]['pnl'].mean() if (df['pnl'] > 0).any() else 0,
        'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if (df['pnl'] < 0).any() else 0,
        'largest_win': df['pnl'].max(),
        'largest_loss': df['pnl'].min(),
    }

    # Profit factor
    total_profit = df[df['pnl'] > 0]['pnl'].sum() if (df['pnl'] > 0).any() else 0
    total_loss = abs(df[df['pnl'] < 0]['pnl'].sum()) if (df['pnl'] < 0).any() else 0
    metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0

    # Trade duration (if columns exist)
    if 'Entry_Time' in df.columns and 'Exit_Time' in df.columns:
        df['Duration'] = pd.to_datetime(df['Exit_Time']) - pd.to_datetime(df['Entry_Time'])
        metrics['avg_trade_duration'] = df['Duration'].mean().total_seconds() / 3600  # hours
    else:
        metrics['avg_trade_duration'] = 0

    # Consecutive losses
    df['is_loss'] = df['pnl'] < 0
    df['loss_streak'] = df['is_loss'].groupby((df['is_loss'] != df['is_loss'].shift()).cumsum()).cumsum()
    metrics['consecutive_losses_max'] = df['loss_streak'].max()

    return metrics

def calculate_composite_score(interview_metrics, trade_metrics):
    """Calculate a comprehensive score combining all metrics."""

    # Weight components
    weights = {
        'win_rate': 0.25,
        'profit_factor': 0.20,
        'roi': 0.20,
        'diversity': 0.15,
        'max_dd': 0.10,
        'consistency': 0.10
    }

    # Normalize metrics to 0-10 scale
    win_rate_score = min(interview_metrics.get('win_rate_interview', 0) / 10, 10)

    pf = trade_metrics.get('profit_factor', 0)
    profit_factor_score = min(pf * 5, 10) if pf > 0 else 0

    # ROI from RANKING (if available)
    roi_score = 0  # Will be added from RANKING.csv

    # Diversity (0.95-1.0 is ideal)
    diversity = interview_metrics.get('diversity_score', 0)
    diversity_score = min(diversity * 10, 10)

    # Max DD (lower is better, penalize > 8%)
    max_dd = interview_metrics.get('max_dd', 0)
    dd_score = max(10 - max_dd * 1.5, 0)

    # Consistency (low consecutive losses)
    consec_losses = trade_metrics.get('consecutive_losses_max', 0)
    consistency_score = max(10 - consec_losses, 0)

    total_score = (
        weights['win_rate'] * win_rate_score +
        weights['profit_factor'] * profit_factor_score +
        weights['roi'] * roi_score +
        weights['diversity'] * diversity_score +
        weights['max_dd'] * dd_score +
        weights['consistency'] * consistency_score
    )

    return {
        'total_score': total_score,
        'win_rate_score': win_rate_score,
        'profit_factor_score': profit_factor_score,
        'diversity_score': diversity_score,
        'dd_score': dd_score,
        'consistency_score': consistency_score
    }

def main():
    print("="*80)
    print("COMPREHENSIVE CHECKPOINT ANALYSIS")
    print("="*80)
    print()

    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]

    all_results = []

    for ckpt in checkpoints:
        print(f"\n[Analyzing] Checkpoint {ckpt:,} steps...")

        # Interview analysis
        interview_path = INTERPRETABILITY_PATH / f"interview_report_{ckpt}.txt"
        interview_metrics = extract_interview_metrics(interview_path)

        # Trades analysis
        trades_path = CHECKPOINTS_ANALYSIS_PATH / f"checkpoint_{ckpt}_trades.csv"
        trade_metrics = analyze_trades_csv(trades_path)

        # Composite score
        scores = calculate_composite_score(interview_metrics, trade_metrics)

        # Combine all metrics
        result = {
            'checkpoint': ckpt,
            **interview_metrics,
            **trade_metrics,
            **scores
        }

        all_results.append(result)

        print(f"   Win Rate: {interview_metrics['win_rate_interview']:.1f}%")
        print(f"   Diversity: {interview_metrics['diversity_score']:.3f}")
        print(f"   Max DD: {interview_metrics['max_dd']:.2f}%")
        print(f"   Profit Factor: {trade_metrics.get('profit_factor', 0):.2f}")
        print(f"   Score: {scores['total_score']:.2f}/10")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by total score
    df = df.sort_values('total_score', ascending=False)

    # Save full report
    output_path = CHECKPOINTS_ANALYSIS_PATH / "COMPREHENSIVE_ANALYSIS.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[Saved] {output_path}")

    # Print top 3
    print("\n" + "="*80)
    print("TOP 3 CHECKPOINTS (Comprehensive Analysis)")
    print("="*80)

    for idx, row in df.head(3).iterrows():
        print(f"\n🏆 RANK #{df.index.get_loc(idx) + 1}: Checkpoint {int(row['checkpoint']):,} steps")
        print(f"   Total Score: {row['total_score']:.2f}/10")
        print(f"   Win Rate: {row['win_rate_interview']:.1f}%")
        print(f"   Profit Factor: {row['profit_factor']:.2f}")
        print(f"   Max DD: {row['max_dd']:.2f}%")
        print(f"   Diversity: {row['diversity_score']:.3f}")
        print(f"   HOLD %: {row['hold_pct']:.1f}%")
        print(f"   Trades: {int(row['num_trades'])}")
        print(f"   Consecutive Losses: {int(row['consecutive_losses_max'])}")

    print("\n" + "="*80)
    print(f"[OK] Comprehensive analysis complete!")
    print(f"[OK] Full report: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
