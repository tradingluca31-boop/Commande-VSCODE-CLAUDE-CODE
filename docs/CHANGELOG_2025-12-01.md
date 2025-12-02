# CHANGELOG - 01 Décembre 2025

## 📋 Modifications Apportées

**Date** : 01-12-2025
**Auteur** : Claude Code
**Objectif** : Ajout de documentation complète pour mémoire permanente Claude Code

---

## 🆕 Nouveaux Fichiers Ajoutés

### 1. `docs/CLAUDE_CODE_REFERENCE.md`
**Description** : Référence permanente complète pour toutes les instances Claude Code
**Contenu** :
- Structure complète du repository
- Architecture réseau détaillée (RecurrentPPO + LSTM)
- 229 features expliquées (209 base + 13 RL + 7 mémoire)
- Hyperparamètres V2.1 complets
- Fonction de récompense hiérarchique (3 tiers)
- Risk Management FTMO compliant
- Callbacks & Monitoring (Checkpoint + Interpretability)
- Tests & Validation (MINI + FULL)
- Analyse SHAP détaillée
- Configuration & Données
- Features avancées (Trade Quality Memory, Missed Opportunities)
- Historique des versions (V2.1 vs V2.0)
- Objectifs & KPI cibles

**Pourquoi ce fichier ?**
- Permet à chaque instance Claude Code de comprendre EXACTEMENT l'architecture de l'agent
- Documentation centralisée pour éviter les erreurs de modification
- Référence pour respecter les règles MODIFY_ONLY et NO_DUPLICATION

---

### 2. `docs/AGENT_7_CONFIG.json`
**Description** : Configuration structurée en JSON pour parsing programmatique
**Contenu** :
- `agent_info` : Informations générales (version 2.1, repository, type)
- `rules_critical` : Règles GITHUB_ONLY, MODIFY_ONLY, NO_DUPLICATION
- `repository_structure` : Tous les dossiers et fichiers avec flag modify_only
- `architecture` : Actor/Critic [256,256], LSTM 256, policy MlpLstmPolicy
- `hyperparameters_v2_1` : Learning rate, training params, PPO specifics, entropy schedule
- `features` : 229 features détaillées (base + RL + memory)
- `reward_function` : Structure hiérarchique avec formules JSON
- `risk_management` : Règles FTMO, dynamic multiplier, position sizing
- `advanced_features` : Trade Quality Memory, Missed Opportunities, Advanced Risk Metrics
- `environment_config` : Balance, action/observation space, data periods
- `constants` : Spread, slippage, commission, ATR multiplier, etc.
- `callbacks_details` : Checkpoint Evaluation + Interpretability (6 questions)
- `testing` : MINI (100 steps) + FULL (1000 steps) avec critères de succès
- `shap_analysis` : 500 samples, visualisations, détection mode collapse
- `performance_targets` : KPIs avec priorités (Sharpe ≥2.5, DD <6%, etc.)
- `version_history` : V2.1 (current) vs V2.0 (deprecated)
- `modification_workflow` : 5 étapes + actions interdites

**Pourquoi ce fichier ?**
- Permet le parsing automatique par scripts Python/JavaScript
- Structure parsable pour outils d'analyse automatique
- Configuration lisible par machine ET humain

---

### 3. `docs/CHANGELOG_2025-12-01.md`
**Description** : Ce fichier - Historique détaillé des modifications du 01-12-2025

**Pourquoi ce fichier ?**
- Trace claire de TOUTES les modifications apportées
- Permet aux futures instances Claude Code de comprendre l'évolution
- Documentation des changements pour l'équipe

---

## 🎯 Règles Critiques Documentées

### Règle #1 : MODIFY_ONLY (Ne jamais créer de versions)
**Problème résolu** : Éviter la duplication de fichiers et la confusion
**Application** :
- Tous les fichiers Python existants doivent être MODIFIÉS, jamais dupliqués
- Un seul `train_CRITIC_BOOST_LSTM.py`, pas de `train_v2.py` ou `train_backup.py`
- Un seul `trading_env_v2_ultimate.py`, pas de versions multiples

### Règle #2 : GITHUB_ONLY (Jamais travailler en local)
**Problème résolu** : Source de vérité unique sur GitHub
**Application** :
- Toujours lire depuis GitHub via WebFetch raw URLs
- Modifications commitées et pushées immédiatement
- Pas de travail "offline" sur AGENT-7

### Règle #3 : NO_DUPLICATION (Économie mémoire)
**Problème résolu** : Gaspillage mémoire et confusion des versions
**Application** :
- Un fichier = une version modifiable
- Historique géré par Git, pas par nommage de fichiers

---

## 📊 Architecture Documentée

### RecurrentPPO + LSTM V2.1
**Caractéristiques principales** :
- **Actor** : [256, 256] neurones (separate independent)
- **Critic** : [256, 256] neurones (separate independent)
- **LSTM** : 256 neurons, 1 layer, 16 steps memory
- **Value Coefficient** : 1.0 (BOOSTED de 0.25 en V2.0) ← **FIX CRITIQUE**
- **Epochs** : 25 (augmenté de 15 en V2.0)
- **Entropie** : Schedule adaptatif 0.20→0.05

### Features (229 Total)
**Décomposition** :
- 209 Base Features (indicateurs techniques)
- 13 RL Features (last action, regret, position, market regime, etc.)
- 7 Memory Features **(NOUVEAU en V2.1)** :
  - Recent Win Rate (20 last trades)
  - Win/Loss Streak
  - Average PnL (20 last)
  - Best Trade (20 last)
  - Worst Trade (20 last)
  - Win Count (20 last)
  - Loss Count (20 last)

---

## 🛡️ Risk Management FTMO Compliant

### Règles Documentées
| Règle | Seuil | Comportement |
|-------|-------|--------------|
| Max Risk/Trade | 1.0% | Kelly optimized 0.33%-1.0% |
| Daily Loss Limit | 2% | Block trades/terminate episode |
| Max Drawdown | 10% | Terminate immediately |
| Emergency Stop (Training) | 20% | Force-close |
| Emergency Stop (Production) | 9.5% | Force-close |
| Risk Reduction Zone | 7%-10% DD | Progressive multiplier 1.0× → 0.0× |
| Recovery Threshold | <5.5% DD | Resume normal risk |

---

## 🔍 Callbacks Documentés

### CheckpointEvaluationCallback
- **Fréquence** : 50,000 steps
- **Outputs** :
  - `checkpoint_[steps]_stats.csv`
  - `checkpoint_[steps]_trades.csv`
  - `RANKING.csv` + `RANKING.txt`

### InterpretabilityCallback
- **Fréquence** : 50,000 steps
- **Format** : Interview 6 questions
- **Output** : `interview_report_[steps].txt`

---

## 🧪 Tests Documentés

### Smoke Test MINI
- **Duration** : ~1 minute
- **Steps** : 100
- **Checks** : Action Distribution + Position Management

### Smoke Test FULL
- **Duration** : ~10 minutes
- **Steps** : 1000
- **Validation** : 4 phases (Model Loading, Action Diversity, Position Management, Stability)

---

## 📈 Analyse SHAP Documentée

- **Samples** : 500 états
- **Background** : 100 échantillons
- **Visualisations** :
  - Global importance
  - Summary plot
  - Waterfall SELL
  - Waterfall BUY
  - Report texte

---

## 🎯 Performance Targets Documentés

| Métrique | Objectif | Priorité |
|----------|----------|----------|
| Sharpe Ratio | ≥2.5 | ⭐⭐⭐ CRITICAL |
| Max Drawdown | <6% | ⭐⭐⭐ CRITICAL (FTMO) |
| ROI | 18-22% | ⭐⭐⭐ CRITICAL |
| Win Rate | >50% | ⭐⭐ HIGH |
| Profit Factor | ≥1.5 | ⭐⭐ HIGH |
| Action Balance | 20-40% HOLD | ⭐ MEDIUM |
| Critic Std | >1.0 | ⭐⭐⭐ CRITICAL |

---

## 📝 Fichiers à Modifier (JAMAIS dupliquer)

Liste complète des fichiers modify_only :
1. `training/train_CRITIC_BOOST_LSTM.py`
2. `environment/trading_env_v2_ultimate.py`
3. `callbacks/CheckpointEvaluationCallback.py`
4. `callbacks/InterpretabilityCallback.py`
5. `tests/smoke_test_MINI.py`
6. `tests/smoke_test_agent7.py`
7. `analysis/explain_shap_agent7.py`

---

## 🔄 Workflow de Modification Documenté

### 5 Étapes Standard
1. **Read file from GitHub** (WebFetch raw URL)
2. **Identify section to modify** (Analyze code structure)
3. **Apply precise modification** (Preserve existing architecture)
4. **Save modified version** (SAME file, no duplication)
5. **Document change** (Detailed commit message)

### Actions Interdites
❌ Creating new file versions
❌ Duplicating existing files
❌ Working on local copies instead of GitHub
❌ Breaking FTMO compatibility
❌ Modifying 229 features count without documentation

---

## 📚 Resources Documentés

- **Repository** : https://github.com/tradingluca31-boop/AGENT-7-UNIQUEMENT
- **Algorithm Library** : Stable-Baselines3
- **Algorithm** : RecurrentPPO
- **Policy** : MlpLstmPolicy
- **SHAP Methodology** : "A Unified Approach to Interpreting Model Predictions"

---

## 🔮 Impact de ces Modifications

### Pour les futures instances Claude Code
✅ Compréhension complète de l'architecture dès le premier message
✅ Respect automatique des règles MODIFY_ONLY et NO_DUPLICATION
✅ Référence centralisée pour toutes les modifications
✅ Réduction des erreurs de modification
✅ Consistance entre instances

### Pour l'équipe
✅ Documentation technique complète et à jour
✅ Configuration parsable en JSON pour outils automatiques
✅ Historique clair des modifications
✅ Facilite l'onboarding de nouveaux développeurs

---

## ✅ Checklist de Validation

- [x] `CLAUDE_CODE_REFERENCE.md` créé avec documentation complète
- [x] `AGENT_7_CONFIG.json` créé avec structure parsable
- [x] `CHANGELOG_2025-12-01.md` créé avec historique détaillé
- [x] Règles MODIFY_ONLY clairement documentées
- [x] Règles GITHUB_ONLY clairement documentées
- [x] Règles NO_DUPLICATION clairement documentées
- [x] Architecture V2.1 complètement documentée
- [x] 229 features expliquées
- [x] Risk Management FTMO documented
- [x] Callbacks documentés
- [x] Tests documentés
- [x] SHAP analysis documentée
- [x] Performance targets documentés
- [x] Workflow de modification documenté

---

## 🚀 Prochaines Étapes

1. Commit et push sur GitHub
2. Vérifier que les fichiers sont bien visibles sur le repository
3. Tester la lecture depuis GitHub (WebFetch)
4. Valider que les futures instances peuvent parser le JSON

---

**🤖 Note** : Ces modifications NE CHANGENT PAS le code de l'agent. Elles ajoutent uniquement de la documentation pour améliorer la mémoire permanente de Claude Code et faciliter les futures modifications.

**Status** : ✅ Documentation Complete - Ready for Commit & Push
