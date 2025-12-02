# 📋 RÉFÉRENCE PERMANENTE - AGENT 7 UNIQUEMENT

**Repository GitHub** : https://github.com/tradingluca31-boop/AGENT-7-UNIQUEMENT

**⚠️ RÈGLES CRITIQUES POUR CLAUDE CODE :**
1. ✅ **MODIFIER UNIQUEMENT** - Ne jamais créer de nouveaux fichiers/versions
2. ✅ **UN SEUL FICHIER par type** - Pas de duplication
3. ✅ **TRAVAIL SUR GITHUB** - Pas en local, toujours via GitHub
4. ✅ **ÉCONOMIE MÉMOIRE** - Un fichier = une version modifiable

---

## 🏗️ STRUCTURE DU REPOSITORY

```
AGENT-7-UNIQUEMENT/
├── analysis/              # Analyse SHAP et explicabilité
│   └── explain_shap_agent7.py
├── callbacks/             # Callbacks d'entraînement
│   ├── CheckpointEvaluationCallback.py
│   └── InterpretabilityCallback.py
├── docs/                  # Documentation
├── environment/           # Environnement RL
│   └── trading_env_v2_ultimate.py
├── launchers/             # Scripts batch de lancement
│   ├── LAUNCH_TRAINING_500K.bat
│   └── RUN_SMOKE_TEST_MINI.bat
├── tests/                 # Tests de validation
│   ├── smoke_test_MINI.py
│   └── smoke_test_agent7.py
├── training/              # Scripts d'entraînement
│   └── train_CRITIC_BOOST_LSTM.py
└── .gitignore
```

---

## 🎯 DESCRIPTION DE L'AGENT

**Nom** : Agent 7 V2.1 - CRITIC BOOST + LSTM
**Type** : Agent de trading RL (Reinforcement Learning)
**Asset** : XAUUSD (Or/USD) - Timeframe H1 (1 heure)
**Algorithme** : RecurrentPPO avec LSTM (256 neurones, mémoire 16 steps)

**Objectifs de performance** :
- Sharpe Ratio : 2.5+
- Max Drawdown : <6%
- ROI : 18-22%

**Version actuelle** : V2.1
**Problème résolu** : "Critic Flat" de la V2.0 (Std Critique passait de 1.0→0.05)

---

## 🧠 ARCHITECTURE RÉSEAU

### Réseau Principal
- **Politique** : MlpLstmPolicy
- **Actor** : [256, 256] neurones - Architecture séparée
- **Critic** : [256, 256] neurones - Architecture séparée et indépendante
- **LSTM** : 256 neurones cachés, 1 couche, mémoire 16 steps

### Hyperparamètres Clés (V2.1)

| Paramètre | Valeur | Notes |
|-----------|--------|-------|
| Learning Rate | 5e-6 → 1e-5 | Adaptatif selon progression |
| Total Steps | 500,000 | ~2 heures d'entraînement |
| Batch Size | 64 | |
| Rollout Length | 2,048 steps | |
| Epochs per Update | 25 | ↑ de 15 (V2.0) |
| Gamma (discount) | 0.9549945651081264 | |
| GAE Lambda | 0.95 | Ajusté de 0.9576 |
| Clipping Range | 0.2 | |
| Max Gradient Norm | 0.5 | |
| **Value Coefficient** | **1.0** | 🔥 Boosted de 0.25 (V2.0) |
| Entropy Coefficient | 0.20 → 0.05 | Adaptatif |

### Entropie Adaptative (Schedule)
- **Phase 1 (0-30%)** : entropy = 0.20 (exploration)
- **Phase 2 (30-70%)** : decay exponentiel (rate 2.5)
- **Phase 3 (70-100%)** : entropy = 0.05 (exploitation)

---

## 📊 FEATURES (229 TOTAL)

### Features de Base (209)
Indicateurs techniques classiques :
- **Oscillateurs** : RSI_14, Stochastic_K, Williams_R, CCI, MFI, ROC
- **Tendance** : MACD, MACD_Signal, ADX, Trend_Strength
- **Volatilité** : ATR_14, BB_Width, Volatility_Percentile
- **Volume** : Volume_Ratio
- **+ 196 autres features techniques**

### Features RL-Spécifiques (13)
1. Last Action (one-hot, 3 features) : SELL/HOLD/BUY
2. Regret Signal (1) : opportunités manquées
3. Position Duration Normalized (1)
4. Unrealized PnL Ratio (1)
5. Market Regime (one-hot, 3 features) : ranging/trending/volatile
6. Hours Until Macro Event (1)
7. Volatility Percentile (1)
8. Position Side Normalized (1)
9. Trade Similarity Score (1) : pattern recognition vs gagnants/perdants historiques

### Features Mémoire/Critic Boost (7)
10. Recent Win Rate (20 derniers trades)
11. Win/Loss Streak
12. Average PnL (20 derniers trades)
13. Best Trade (20 derniers)
14. Worst Trade (20 derniers)
15. Win Count (20 derniers)
16. Loss Count (20 derniers)

**Total : 209 + 13 + 7 = 229 features**

---

## 💰 FONCTION DE RÉCOMPENSE (Hiérarchique)

### TIER 1 - Trading Core (70%)
- **Profit** (40%) : `(equity - initial_balance) / initial_balance`
- **Sharpe Ratio** (20%) : `returns.mean() / returns.std() × √252`
- **Drawdown Penalty** (10%) : `-max_drawdown × 5.0`

### TIER 2 - Risk Management (20%)
- **FTMO Compliance** (10%) :
  - 1.0 si DD < 10%
  - 0.5 si DD entre 7-10%
  - 0.0 si violé
- **VaR 95% Management** (5%) : pénalité selon VaR absolu
- **Tail Risk Control** (5%) : détection kurtosis (>3.0)

### TIER 3 - Behavioral Shaping (10%)
Bonus V2 :
- Direction Prediction : +0.02
- Profit Taking : +0.05 à +0.20
- Loss Cutting : +0.03
- Completion : +0.10

**Multiplicateur Adaptatif** : [0.5–2.0] selon performance récente

**Reward Final** = (Tier1 + Tier2 + Tier3) × Adaptive_Multiplier

---

## 🛡️ RISK MANAGEMENT (FTMO COMPLIANT)

### Paramètres de Risque

| Règle | Seuil | Comportement |
|-------|-------|--------------|
| Max Risk Per Trade | 1.0% | Sizing Kelly optimisé : 0.33%–1.0% adaptatif |
| Daily Loss Limit | 2% | Bloque nouveaux trades ou termine l'épisode |
| Max Drawdown | 10% | Termine l'épisode immédiatement |
| Emergency Stop (Training) | 20% | Force-close position |
| Emergency Stop (Production) | 9.5% | Force-close position |
| Risk Reduction Zone | 7%–10% DD | Multiplicateur progressif : 1.0× → 0.0× |
| Recovery Threshold | <5.5% DD | Réactivation exposition normale |

### Dynamic Risk Multiplier
```
Si DD < 7%         : multiplier = 1.0   (100% risque normal)
Si 7% ≤ DD < 10%   : multiplier = max(0, 1.0 - (DD - 7%)/3%)
Si DD ≥ 10%        : multiplier = 0.0   (pas de trading)
```

### Position Sizing Formula
```python
Risk_Amount = Balance × Base_Risk × Confidence × Risk_Multiplier
Position_Size = Risk_Amount / (ATR × ATR_Multiplier × Contract_Size)
```

### Confidence Threshold (Sélectivité Dynamique)
- **Base** : 60% (normal) ou 70% (DD ≥ 7%)
- **Modulation** : ±5% selon Trade Similarity Score
- **Résultat** : seuil 5%–85% (permissif en entraînement)

---

## 🔍 CALLBACKS & MONITORING

### 1. CheckpointEvaluationCallback

**Fréquence** : Tous les 50,000 steps
**Durée** : ~10 secondes (<0.1% overhead)

**Métriques trackées** :
- **Financières** : Initial/Final Balance, Total P&L, ROI%, Max Drawdown%
- **Trading** : Nombre trades, Win Rate, Profit Factor
- **Trades détaillés** : Entry/Exit price, Size, P&L, Side

**Outputs** :
- `checkpoint_[steps]_stats.csv` : Métriques agrégées
- `checkpoint_[steps]_trades.csv` : Trade-by-trade records
- `RANKING.csv` + `RANKING.txt` : Classement checkpoints (score 0-10)

### 2. InterpretabilityCallback

**Fréquence** : Tous les 50,000 steps
**Format** : "Interview" de l'agent en 6 questions

**Questions posées** :
1. **Feature Importance** : Perturbation analysis (50 états)
2. **Action Patterns** : Distribution SELL/HOLD/BUY sur 100 scénarios
   - Diagnostic : Under-trading (>60% HOLD) ou Over-trading (<20% HOLD)
3. **Market Regime Response** : Adaptation ranging/trending/volatile
4. **Trade Triggers** : Conditions précédant actions (regime, DD, position)
5. **Risk Management** : Max DD, Risk Multiplier, Kelly, VaR 95%
6. **Error Analysis** : Trades perdants, taux perte, sévérité

**Output** : `interview_report_[steps].txt`

---

## 🧪 TESTS & VALIDATION

### 1. Smoke Test MINI (`smoke_test_MINI.py`)

**Durée** : ~1 minute
**Steps** : 100
**Objectif** : Validation rapide fonctionnalité de base

**Checks** :
1. Data Loading
2. Environment Creation
3. Model Loading (5 chemins de fallback)
4. **Action Distribution** :
   - ❌ FAIL si une action >90% ("MODE COLLAPSE")
   - ⚠️ WARNING si max action >70%
   - ✅ PASS si ≤70% max
5. **Position Management** :
   - Doit ouvrir ET fermer des positions
   - Fail si jamais d'ouverture ou jamais de fermeture

**Critères de succès** : Distribution équilibrée + Gestion positions active

### 2. Smoke Test FULL (`smoke_test_agent7.py`)

**Durée** : ~10 minutes
**Steps** : 1000
**Checkpoints** : Tous les 100 steps (10 total)

**Validation (4 phases)** :
1. **[1/4] Model Loading** : Compatibilité checkpoint/environnement
2. **[2/4] Action Diversity** : Même logic que MINI
3. **[3/4] Position Management** : Entrée/sortie marché vérifiées
4. **[4/4] Stability** : Pas de crash sur 1000 steps

**Critères de succès** : "Agent ready for full training/evaluation"

---

## 📈 ANALYSE SHAP (`explain_shap_agent7.py`)

**Scope** : 222 features, 3 actions (SELL/HOLD/BUY)
**Échantillons** : 500 états collectés
**Background Dataset** : 100 échantillons pour KernelExplainer

### Méthode SHAP
- **Explainer** : `shap.KernelExplainer`
- **Fonction** : Probabilités actions du PPO policy network → (N, 3)

### Importance Globale
- Mean Absolute SHAP values par feature
- Top-5 et Top-10 cumulative influence

### Détection
- **Mode Collapse** : Si action unique >80%
- **Cancellation Effects** : Paires features avec SHAP opposés

### Visualisations Générées

| Fichier | Description |
|---------|-------------|
| `shap_global_importance.png` | Top 20 features (mean abs SHAP) |
| `shap_summary_plot.png` | Distribution 222 features (top 30) |
| `shap_waterfall_SELL.png` | Décomposition meilleure décision SELL |
| `shap_waterfall_BUY.png` | Décomposition meilleure décision BUY |
| `shap_report.txt` | Rapport textuel complet + diagnostics |

---

## 🔧 CONFIGURATION & DONNÉES

### Environnement
- **Initial Balance** : $100,000
- **Action Space** : Discrete(3) ou Continuous Box(2,)
- **Observation Space** : Box(229,) normalisé
- **Data Frequency** : Chandeliers 1 heure (H1)

### Données
- **Training** : 2008-2020
- **Validation** : 2021
- **Asset** : XAUUSD

### Constantes (config.py)
```python
SPREAD_PIPS = 0.3              # Spread typique Gold
SLIPPAGE_PIPS = 0.2
COMMISSION_PER_LOT = 25        # USD par lot
ATR_MULTIPLIER = 2.0
XAUUSD_CONTRACT_SIZE = 100     # oz
XAUUSD_PIP_VALUE = 0.01        # USD
TRAINING_MODE = True           # Enable 20% emergency DD
```

---

## 📝 FEATURES AVANCÉES

### Trade Quality Memory (WALL STREET V3)
- Capture 15 entry features par trade (RSI, MACD, ATR, ADX, etc.)
- Compare contexte actuel avec top 25 winners + 25 losers
- Scoring par similarité cosinus → Opportunités manquées pattern-based

### Missed Opportunities Tracking
- Stocke les HOLDs et évalue après 10 steps si prix a bougé >0.3%
- Pénalise passivité quand trend strength >0.6

### Advanced Risk Metrics
- Kelly Criterion fraction (sizing optimal) [0, 0.5]
- Value at Risk 95% percentile
- Tail risk detection via excess kurtosis (>3.0)

---

## 🚀 LAUNCHERS (Batch Files)

### LAUNCH_TRAINING_500K.bat
Lance l'entraînement complet de 500K steps

### RUN_SMOKE_TEST_MINI.bat
Lance le smoke test rapide (100 steps)

---

## 📌 HISTORIQUE DES VERSIONS

### V2.1 (Actuel) - CRITIC BOOST + LSTM
**Problème résolu** : Critic Flat (Std 1.0→0.05)

**Changements majeurs** :
1. ✅ Value Coefficient : 0.25 → **1.0** (4x boost)
2. ✅ Epochs : 15 → **25** (+67%)
3. ✅ Ajout 7 Memory Features (win rate, streaks, avg PnL, etc.)
4. ✅ Entropie adaptative (0.20→0.05)
5. ✅ Actor/Critic séparés et indépendants

**Résultat attendu** : Critic Std >1.0 stable, convergence saine

### V2.0 (Précédent)
❌ Problème : Critic Flat - Std tombait à 0.05
❌ Cause : Value coefficient trop faible (0.25), manque features mémoire

---

## 🎯 OBJECTIFS & KPI CIBLES

| Métrique | Objectif | Critique |
|----------|----------|----------|
| Sharpe Ratio | ≥2.5 | ⭐⭐⭐ |
| Max Drawdown | <6% | ⭐⭐⭐ FTMO |
| ROI | 18-22% | ⭐⭐⭐ |
| Win Rate | >50% | ⭐⭐ |
| Profit Factor | ≥1.5 | ⭐⭐ |
| Action Balance | 20-40% HOLD | ⭐ (éviter mode collapse) |
| Critic Std | >1.0 | ⭐⭐⭐ Santé apprentissage |

---

## ⚠️ RAPPELS CRITIQUES POUR CLAUDE CODE

### Règles de Modification
1. **JAMAIS** créer de nouveaux fichiers sans raison absolue
2. **TOUJOURS** modifier les fichiers existants
3. **UN SEUL** fichier par type (pas de versions multiples)
4. **GITHUB** comme source de vérité (pas local)

### Fichiers à Modifier (JAMAIS dupliquer)
- `training/train_CRITIC_BOOST_LSTM.py` → Script entraînement
- `environment/trading_env_v2_ultimate.py` → Environnement RL
- `callbacks/CheckpointEvaluationCallback.py` → Évaluation checkpoints
- `callbacks/InterpretabilityCallback.py` → Interviews agent
- `tests/smoke_test_MINI.py` → Test rapide
- `tests/smoke_test_agent7.py` → Test complet
- `analysis/explain_shap_agent7.py` → Analyse SHAP

### Workflow de Modification Recommandé
1. 📖 Lire fichier depuis GitHub (WebFetch raw)
2. 🔍 Identifier section à modifier
3. ✏️ Appliquer modification précise
4. 💾 Sauvegarder version modifiée (MÊME fichier)
5. 📝 Documenter changement dans commit message

---

## 📚 RESSOURCES & RÉFÉRENCES

- **Repository** : https://github.com/tradingluca31-boop/AGENT-7-UNIQUEMENT
- **Algorithm** : RecurrentPPO (Stable-Baselines3)
- **Policy** : MlpLstmPolicy
- **Méthodologie SHAP** : "A Unified Approach to Interpreting Model Predictions"

---

## 🔄 DERNIÈRE MISE À JOUR

**Date** : 2025-12-01
**Version Agent** : V2.1 - CRITIC BOOST + LSTM
**Commit** : "Add comprehensive documentation for Claude Code memory (01-12-2025)"
**Status** : ✅ Production Ready

---

**🤖 Note pour les instances Claude Code futures** :
Ce document est la RÉFÉRENCE PERMANENTE. Toutes les modifications de l'AGENT-7 doivent :
- Respecter l'architecture existante
- Modifier les fichiers (ne pas créer de versions)
- Être documentées ici si changements majeurs
- Préserver la compatibilité FTMO et les 229 features

**Économisez la mémoire. Un fichier = une version. Modifiez, ne dupliquez pas.**
