# -*- coding: utf-8 -*-
"""
SHAP EXPLAINABILITY - AGENT 7 V2 ULTIMATE

Feature-Level Explainability pour comprendre EXACTEMENT:
- Quelle feature pousse vers SELL/HOLD/BUY?
- Pourquoi mode collapse sur une action?
- Comment l'agent combine les 222 features?
- Quelles features il ignore (poids faible)?
- Features contradictoires qui s'annulent?

Standard: Renaissance Technologies, Two Sigma, Citadel
Paper: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"

Usage:
    python explain_shap_agent7.py

Output:
    - shap_global_importance.png (top 20 features)
    - shap_summary_plot.png (vue d'ensemble 222 features)
    - shap_waterfall_SELL.png (exemple décision SELL)
    - shap_waterfall_BUY.png (exemple décision BUY)
    - shap_report.txt (rapport textuel détaillé)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from stable_baselines3 import PPO
from collections import Counter

# Add paths
script_dir = Path(__file__).resolve().parent
agent7_dir = script_dir.parent
agent_v2_dir = agent7_dir.parent
project_root = agent_v2_dir.parent
sys.path.insert(0, str(agent_v2_dir))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from trading_env_v2_ultimate import GoldTradingEnv
import config

print("="*80)
print("SHAP EXPLAINABILITY - AGENT 7 V2 ULTIMATE")
print("="*80)
print()
print("Feature-Level Analysis pour comprendre COMMENT l'agent pense...")
print("Duree: ~5-10 minutes")
print()

# ============================================================================
# 1. LOAD DATA & MODEL
# ============================================================================
print("[1/5] Loading data & model...")

# Load data
loader = DataLoader(verbose=False)
aligned_df, auxiliary_data = loader.load_all_data()

# Training period
train_start = config.TRAIN_START_DATE
train_end = config.TRAIN_END_DATE
aligned_df_train = aligned_df.loc[train_start:train_end]

# Extract prices (H1 for Agent 7)
df_h1_raw = auxiliary_data['xauusd_raw']['H1']
df_h1_raw = df_h1_raw.loc[train_start:train_end]
df_prices = df_h1_raw[['open', 'high', 'low', 'close', 'volume']].copy()

# Calculate features
fe = FeatureEngineering(aligned_df_train, auxiliary_data, verbose=False)
df_features = fe.compute_all_features(
    start_date=train_start,
    end_date=train_end
)

# Create environment
env = GoldTradingEnv(
    prices_df=df_prices,
    features_df=df_features,
    initial_balance=config.INITIAL_BALANCE,
    action_space_type='discrete',
    verbose=False
)

# Get feature names (222 total)
feature_names = df_features.columns.tolist()
rl_feature_names = [
    'action_last_sell', 'action_last_hold', 'action_last_buy',
    'regret_signal', 'position_duration', 'unrealized_pnl_ratio',
    'regime_ranging', 'regime_trending', 'regime_volatile',
    'hours_until_event', 'volatility_percentile', 'position_side',
    'trade_similarity'
]
all_feature_names = feature_names + rl_feature_names

# Load model
models_dir = agent7_dir / 'models'
checkpoint_paths = [
    models_dir / 'checkpoints' / 'agent7_ultimate_500000_steps.zip',
    models_dir / 'checkpoints' / 'agent7_ultimate_450000_steps.zip',
    models_dir / 'checkpoints' / 'agent7_ultimate_400000_steps.zip',
    models_dir / 'best_model.zip'
]

model = None
for checkpoint_path in checkpoint_paths:
    if checkpoint_path.exists():
        try:
            # Load model without env (we have our own)
            model = PPO.load(checkpoint_path)
            print(f"[OK] Model loaded: {checkpoint_path.name}")
            break
        except:
            continue

if model is None:
    print("[ERROR] No checkpoint found")
    sys.exit(1)

print()

# ============================================================================
# 2. COLLECT SAMPLES FOR SHAP ANALYSIS
# ============================================================================
print("[2/5] Collecting samples (500 states)...")

samples = []
obs = env.reset()

# Extract observation if it's a tuple/dict
if isinstance(obs, tuple):
    obs = obs[0]  # Get first element (observation)
if isinstance(obs, dict):
    obs = obs['observation'] if 'observation' in obs else list(obs.values())[0]

for step in range(500):
    # Store observation (ensure it's numpy array)
    obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
    samples.append({
        'obs': obs_array.copy(),
        'step': step
    })

    # Take action
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        obs = env.reset()
        # Extract observation again after reset
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            obs = obs['observation'] if 'observation' in obs else list(obs.values())[0]

print(f"[OK] Collected {len(samples)} samples")
print()

# Create observation matrix
X = np.array([s['obs'] for s in samples])  # (500, 222)

print(f"[DEBUG] Observation matrix shape: {X.shape}")
print(f"[DEBUG] Feature names: {len(all_feature_names)}")
print()

# ============================================================================
# 3. CREATE SHAP EXPLAINER
# ============================================================================
print("[3/5] Creating SHAP explainer...")
print("     (This may take 2-3 minutes for 500 samples)")
print()

# Define prediction function for SHAP
def predict_proba(observations):
    """
    Predict action probabilities for SHAP

    Args:
        observations: numpy array (N, 222)

    Returns:
        probabilities: numpy array (N, 3) - [P(SELL), P(HOLD), P(BUY)]
    """
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observations).float()
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()
    return probs

# Create explainer (use a background of 100 samples for speed)
background = X[np.random.choice(X.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(predict_proba, background)

# Calculate SHAP values (this is slow - 500 samples)
print("     Computing SHAP values (patience...)...")
shap_values = explainer.shap_values(X)

# shap_values is a list of 3 arrays (one per action):
# shap_values[0] = SHAP for SELL (500, 222)
# shap_values[1] = SHAP for HOLD (500, 222)
# shap_values[2] = SHAP for BUY (500, 222)

print("[OK] SHAP values computed")
print()

# DEBUG: Check SHAP values structure
print("[DEBUG] Type of shap_values:", type(shap_values))
if isinstance(shap_values, list):
    print("[DEBUG] Number of arrays:", len(shap_values))
    print("[DEBUG] Shape of first array:", shap_values[0].shape)
else:
    print("[DEBUG] Shape of shap_values:", shap_values.shape)
print("[DEBUG] Number of feature names:", len(all_feature_names))
print()

# ============================================================================
# 4. ANALYZE GLOBAL FEATURE IMPORTANCE
# ============================================================================
print("[4/5] Analyzing global feature importance...")

# Calculate mean absolute SHAP value for each feature (for SELL)
# SHAP returns list of 3 arrays (one per action)
if isinstance(shap_values, list):
    shap_sell = shap_values[0]  # (500, 222)
else:
    # If single array, assume it's for the dominant action
    shap_sell = shap_values

mean_shap_abs = np.abs(shap_sell).mean(axis=0)  # (222,)
print(f"[DEBUG] mean_shap_abs shape: {mean_shap_abs.shape}")
print()

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'feature': all_feature_names,
    'shap_importance': mean_shap_abs
}).sort_values('shap_importance', ascending=False)

print("Top 20 Most Important Features (for SELL decisions):")
print("-" * 80)
for i, row in feature_importance.head(20).iterrows():
    print(f"{i+1:2d}. {row['feature']:45s} {row['shap_importance']:.4f}")

print()

# ============================================================================
# 5. CREATE VISUALIZATIONS
# ============================================================================
print("[5/5] Creating visualizations...")

output_dir = agent7_dir / 'models' / 'shap_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

# Visualization 1: Global Feature Importance (Top 20)
plt.figure(figsize=(12, 8))
top20 = feature_importance.head(20)
plt.barh(range(len(top20)), top20['shap_importance'])
plt.yticks(range(len(top20)), top20['feature'])
plt.xlabel('Mean |SHAP value| (Average Impact on Output)')
plt.title('Top 20 Most Important Features for SELL Decisions')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'shap_global_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: shap_global_importance.png")

# Visualization 2: Summary Plot (All features)
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_sell,
    X,
    feature_names=all_feature_names,
    show=False,
    max_display=30
)
plt.title('SHAP Summary Plot - SELL Decisions (Top 30 Features)')
plt.tight_layout()
plt.savefig(output_dir / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: shap_summary_plot.png")

# Visualization 3: Waterfall Plot (Example SELL decision)
# Find sample with highest SELL probability
sell_probs = []
for i in range(len(samples)):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(samples[i]['obs']).unsqueeze(0).float()
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]
        sell_probs.append(probs[0])

max_sell_idx = np.argmax(sell_probs)
print(f"\n  [DEBUG] Example SELL decision: sample #{max_sell_idx}, P(SELL)={sell_probs[max_sell_idx]:.1%}")

plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_sell[max_sell_idx],
        base_values=explainer.expected_value[0],
        data=X[max_sell_idx],
        feature_names=all_feature_names
    ),
    show=False,
    max_display=15
)
plt.title(f'SHAP Waterfall Plot - SELL Decision (P={sell_probs[max_sell_idx]:.1%})')
plt.tight_layout()
plt.savefig(output_dir / 'shap_waterfall_SELL.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: shap_waterfall_SELL.png")

# Visualization 4: Waterfall Plot (Example BUY decision)
buy_probs = []
for i in range(len(samples)):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(samples[i]['obs']).unsqueeze(0).float()
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]
        buy_probs.append(probs[2])

max_buy_idx = np.argmax(buy_probs)
print(f"  [DEBUG] Example BUY decision: sample #{max_buy_idx}, P(BUY)={buy_probs[max_buy_idx]:.1%}")

plt.figure(figsize=(12, 8))
shap_buy = shap_values[2]
shap.waterfall_plot(
    shap.Explanation(
        values=shap_buy[max_buy_idx],
        base_values=explainer.expected_value[2],
        data=X[max_buy_idx],
        feature_names=all_feature_names
    ),
    show=False,
    max_display=15
)
plt.title(f'SHAP Waterfall Plot - BUY Decision (P={buy_probs[max_buy_idx]:.1%})')
plt.tight_layout()
plt.savefig(output_dir / 'shap_waterfall_BUY.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: shap_waterfall_BUY.png")

print()

# ============================================================================
# 6. GENERATE TEXTUAL REPORT
# ============================================================================
print("[6/6] Generating textual report...")

report_path = output_dir / 'shap_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SHAP EXPLAINABILITY REPORT - AGENT 7 V2 ULTIMATE\n")
    f.write("="*80 + "\n\n")

    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {checkpoint_path.name if model else 'N/A'}\n")
    f.write(f"Samples analyzed: {len(samples)}\n")
    f.write(f"Total features: {len(all_feature_names)}\n\n")

    f.write("="*80 + "\n")
    f.write("SECTION 1: GLOBAL FEATURE IMPORTANCE (TOP 30)\n")
    f.write("="*80 + "\n\n")

    f.write("Features sorted by average absolute SHAP value (impact on SELL):\n\n")
    f.write(f"{'Rank':<6} {'Feature':<45} {'SHAP Importance':<15} {'% Total':<10}\n")
    f.write("-"*80 + "\n")

    total_shap = feature_importance['shap_importance'].sum()
    for idx, (i, row) in enumerate(feature_importance.head(30).iterrows(), 1):
        pct = (row['shap_importance'] / total_shap) * 100
        f.write(f"{idx:<6} {row['feature']:<45} {row['shap_importance']:<15.4f} {pct:<10.2f}%\n")

    f.write("\n")
    f.write("INTERPRETATION:\n")
    f.write("-"*80 + "\n")

    top1_feature = feature_importance.iloc[0]
    top1_pct = (top1_feature['shap_importance'] / total_shap) * 100

    if top1_pct > 15:
        f.write(f"⚠️  WARNING: Top feature '{top1_feature['feature']}' dominates ({top1_pct:.1f}%)\n")
        f.write("   This feature has EXCESSIVE influence on decisions.\n")
        f.write("   Risk: Overfitting, single-feature dependency\n")
        f.write("   Solution: Consider removing or re-normalizing this feature\n\n")
    elif top1_pct > 10:
        f.write(f"⚠️  CAUTION: Top feature '{top1_feature['feature']}' has high influence ({top1_pct:.1f}%)\n")
        f.write("   Monitor this feature for potential overfitting\n\n")
    else:
        f.write(f"✓ GOOD: Feature influence is distributed (top feature = {top1_pct:.1f}%)\n")
        f.write("   Agent uses multiple features for decisions\n\n")

    # Check feature concentration
    top5_pct = (feature_importance.head(5)['shap_importance'].sum() / total_shap) * 100
    top10_pct = (feature_importance.head(10)['shap_importance'].sum() / total_shap) * 100

    f.write(f"Feature Concentration:\n")
    f.write(f"  Top 5 features:  {top5_pct:.1f}% of total influence\n")
    f.write(f"  Top 10 features: {top10_pct:.1f}% of total influence\n\n")

    if top5_pct > 60:
        f.write("⚠️  High concentration: Agent relies heavily on few features\n\n")
    elif top5_pct > 40:
        f.write("⚠️  Moderate concentration: Agent uses some features more than others\n\n")
    else:
        f.write("✓ Low concentration: Agent distributes attention across many features\n\n")

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("SECTION 2: MODE COLLAPSE DIAGNOSIS\n")
    f.write("="*80 + "\n\n")

    # Analyze action distribution
    actions_taken = []
    for i in range(len(samples)):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(samples[i]['obs']).unsqueeze(0).float()
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]
            action = np.argmax(probs)
            actions_taken.append(action)

    action_counts = Counter(actions_taken)
    f.write("Action Distribution (500 samples):\n")
    f.write(f"  SELL: {action_counts[0]:3d} ({action_counts[0]/500*100:.1f}%)\n")
    f.write(f"  HOLD: {action_counts[1]:3d} ({action_counts[1]/500*100:.1f}%)\n")
    f.write(f"  BUY:  {action_counts[2]:3d} ({action_counts[2]/500*100:.1f}%)\n\n")

    dominant_action = max(action_counts, key=action_counts.get)
    dominant_pct = action_counts[dominant_action] / 500 * 100

    if dominant_pct > 80:
        action_name = ['SELL', 'HOLD', 'BUY'][dominant_action]
        f.write(f"⚠️  MODE COLLAPSE DETECTED: {action_name} = {dominant_pct:.1f}%\n\n")
        f.write(f"Top 5 features pushing towards {action_name}:\n")
        f.write("-"*80 + "\n")

        # Find features with highest positive SHAP for dominant action
        mean_shap = shap_values[dominant_action].mean(axis=0)
        top_features_idx = np.argsort(np.abs(mean_shap))[-5:][::-1]

        for idx in top_features_idx:
            f.write(f"  - {all_feature_names[idx]:<45} SHAP={mean_shap[idx]:+.4f}\n")

        f.write("\nRECOMMENDATIONS:\n")
        f.write("  1. Check if top features have data leakage or bias\n")
        f.write("  2. Consider removing highly influential features\n")
        f.write("  3. Increase entropy coefficient for more exploration\n")
        f.write("  4. Add diversity penalty in reward function\n\n")
    elif dominant_pct > 60:
        action_name = ['SELL', 'HOLD', 'BUY'][dominant_action]
        f.write(f"⚠️  BIAS DETECTED: {action_name} = {dominant_pct:.1f}%\n")
        f.write("   Agent has preference but not complete collapse\n\n")
    else:
        f.write("✓ BALANCED: No single action dominates\n")
        f.write("   Agent adapts decisions to market context\n\n")

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("SECTION 3: EXAMPLE DECISIONS BREAKDOWN\n")
    f.write("="*80 + "\n\n")

    # Example SELL decision
    f.write("Example 1: SELL Decision\n")
    f.write("-"*80 + "\n")
    f.write(f"Sample #{max_sell_idx}, P(SELL)={sell_probs[max_sell_idx]:.1%}\n\n")

    f.write("Top 10 Contributing Features:\n")
    shap_idx_sell = np.argsort(np.abs(shap_sell[max_sell_idx]))[-10:][::-1]
    for i, idx in enumerate(shap_idx_sell, 1):
        feat_name = all_feature_names[idx]
        shap_val = shap_sell[max_sell_idx][idx]
        feat_val = X[max_sell_idx][idx]
        direction = "→ SELL" if shap_val > 0 else "→ HOLD/BUY"
        f.write(f"  {i:2d}. {feat_name:<40} Value={feat_val:+.3f}, SHAP={shap_val:+.4f} {direction}\n")

    f.write("\n")

    # Example BUY decision
    f.write("Example 2: BUY Decision\n")
    f.write("-"*80 + "\n")
    f.write(f"Sample #{max_buy_idx}, P(BUY)={buy_probs[max_buy_idx]:.1%}\n\n")

    f.write("Top 10 Contributing Features:\n")
    shap_idx_buy = np.argsort(np.abs(shap_buy[max_buy_idx]))[-10:][::-1]
    for i, idx in enumerate(shap_idx_buy, 1):
        feat_name = all_feature_names[idx]
        shap_val = shap_buy[max_buy_idx][idx]
        feat_val = X[max_buy_idx][idx]
        direction = "→ BUY" if shap_val > 0 else "→ SELL/HOLD"
        f.write(f"  {i:2d}. {feat_name:<40} Value={feat_val:+.3f}, SHAP={shap_val:+.4f} {direction}\n")

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("SECTION 4: FEATURE INTERACTIONS\n")
    f.write("="*80 + "\n\n")

    f.write("Features that might cancel each other out:\n\n")

    # Look for features with opposite SHAP values
    mean_shap_sell = shap_sell.mean(axis=0)

    # Find pairs with opposite signs and similar magnitude
    f.write("(Feature pairs with potential cancellation effects)\n")
    f.write("-"*80 + "\n")

    cancellations_found = 0
    for i in range(len(all_feature_names)):
        for j in range(i+1, len(all_feature_names)):
            if mean_shap_sell[i] * mean_shap_sell[j] < 0:  # Opposite signs
                if 0.5 < abs(mean_shap_sell[i] / mean_shap_sell[j]) < 2.0:  # Similar magnitude
                    cancellations_found += 1
                    if cancellations_found <= 10:  # Show top 10
                        f.write(f"  - {all_feature_names[i]:<35} (SHAP={mean_shap_sell[i]:+.4f})\n")
                        f.write(f"    vs {all_feature_names[j]:<35} (SHAP={mean_shap_sell[j]:+.4f})\n\n")

    if cancellations_found == 0:
        f.write("  No significant cancellation effects detected\n\n")
    else:
        f.write(f"\nTotal potential cancellations: {cancellations_found}\n\n")

    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("Visualizations saved:\n")
    f.write(f"  - {output_dir / 'shap_global_importance.png'}\n")
    f.write(f"  - {output_dir / 'shap_summary_plot.png'}\n")
    f.write(f"  - {output_dir / 'shap_waterfall_SELL.png'}\n")
    f.write(f"  - {output_dir / 'shap_waterfall_BUY.png'}\n")

print(f"[OK] Report saved: {report_path}")
print()

print("="*80)
print("SHAP ANALYSIS COMPLETE!")
print("="*80)
print()
print(f"Output directory: {output_dir}")
print()
print("Files generated:")
print(f"  1. shap_global_importance.png - Top 20 features bar chart")
print(f"  2. shap_summary_plot.png - Feature distribution (all 222)")
print(f"  3. shap_waterfall_SELL.png - Example SELL decision breakdown")
print(f"  4. shap_waterfall_BUY.png - Example BUY decision breakdown")
print(f"  5. shap_report.txt - Detailed textual analysis")
print()
print("Next steps:")
print("  1. Open PNG files to visualize feature importance")
print("  2. Read shap_report.txt for detailed diagnosis")
print("  3. If mode collapse detected, check top features for issues")
print()
