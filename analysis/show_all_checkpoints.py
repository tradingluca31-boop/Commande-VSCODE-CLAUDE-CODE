"""
DISPLAY ALL CHECKPOINTS IN TABLE FORMAT
Shows all checkpoint evaluation metrics in a clean table
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_PATH = Path(__file__).parent.parent / "models" / "checkpoints_analysis"
RANKING_CSV = BASE_PATH / "RANKING.csv"

def main():
    print("=" * 120)
    print("ALL CHECKPOINTS EVALUATION - DETAILED TABLE")
    print("=" * 120)
    print()

    # Load data
    df = pd.read_csv(RANKING_CSV)

    # Sort by checkpoint
    df = df.sort_values('checkpoint')

    # Print header
    print(f"{'Step':<10} {'Score':<8} {'ROI %':<10} {'Win Rate %':<12} {'PF':<8} {'Max DD %':<10} {'Trades':<8} {'HOLD %':<8}")
    print("-" * 120)

    # Print each checkpoint
    for _, row in df.iterrows():
        step = f"{int(row['checkpoint']):,}"
        score = f"{row['score']:.2f}"
        roi = f"{row['roi_pct']:.2f}"
        wr = f"{row['win_rate']:.2f}"
        pf = f"{row['profit_factor']:.2f}"
        dd = f"{row['max_drawdown']:.2f}"
        trades = f"{int(row['total_trades'])}"
        hold = f"{row['action_hold_pct']:.1f}"

        print(f"{step:<10} {score:<8} {roi:<10} {wr:<12} {pf:<8} {dd:<10} {trades:<8} {hold:<8}")

    print("-" * 120)
    print()

    # Print best checkpoint
    best = df.loc[df['score'].idxmax()]
    print(f"BEST CHECKPOINT: {int(best['checkpoint']):,} steps (Score: {best['score']:.2f}/10)")
    print(f"  ROI: {best['roi_pct']:.2f}% | Win Rate: {best['win_rate']:.2f}% | Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Max DD: {best['max_drawdown']:.2f}% | Trades: {int(best['total_trades'])}")

    print()
    print("=" * 120)

if __name__ == "__main__":
    main()
