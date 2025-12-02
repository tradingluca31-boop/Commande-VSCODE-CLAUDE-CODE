# CHANGELOG - Agent 7 V2.1 CRITIC BOOST + LSTM

## 2025-12-02 05:17 - Quick Test 10K RÉUSSI

### RÉSULTAT QUICK TEST 10K
```
SELL: 42.0%
HOLD: 14.8%  (was 80-100%)
BUY:  43.2%
Trades ouverts: 4  (was 0)
Trades fermés: 3
```

**VERDICT: SUCCÈS** - Prêt pour training 500K!

---

## 2025-12-02 05:00 - Session de Diagnostic et Fixes

### PROBLÈME IDENTIFIÉ
- Agent ne trade pas (0-4 trades sur 500K steps)
- Critic V(s) = 0.0000 toujours (MORT)
- Mode collapse: 100% HOLD ou 100% SELL

### DIAGNOSTIC EFFECTUÉ

#### Scripts créés pour diagnostic :
| Script | Description |
|--------|-------------|
| `analysis/agent7_parle.py` | Interview automatique de l'agent |
| `analysis/check_policy_values.py` | Check P(SELL), P(HOLD), P(BUY) et V(s) |
| `analysis/check_reward_function.py` | Analyse reward par action |
| `analysis/find_blocker.py` | Trouve ce qui bloque les trades |
| `analysis/diagnostic_complet_v2.py` | Diagnostic complet |

#### Launchers créés :
| Launcher | Description |
|----------|-------------|
| `launchers/RUN_AGENT7_PARLE.bat` | Lance l'interview agent |
| `launchers/RUN_CHECK_POLICY_VALUES.bat` | Lance check policy |
| `launchers/RUN_CHECK_REWARD.bat` | Lance check reward |
| `launchers/RUN_FIND_BLOCKER.bat` | Lance find blocker |
| `launchers/RUN_QUICK_TEST_10K.bat` | Test rapide 10K steps |

### RÉSULTATS DIAGNOSTIC

```
Environment:     OK - trades s'ouvrent quand forcés
Reward function: OK - E[reward|TRADE] > E[reward|HOLD]
Policy (probas): OK - équilibré ~33/33/33
Critic V(s):     MORT - toujours 0.0000
```

**ROOT CAUSE** : Critic mort → agent ne peut pas évaluer les états

---

### FIXES APPLIQUÉS

#### 1. Paramètres Entropy (train_CRITIC_BOOST_LSTM.py)

| Paramètre | Avant | Après |
|-----------|-------|-------|
| `ent_coef` initial | 0.25 | **0.35** |
| `ent_coef` minimum | 0.12 | **0.15** |
| High entropy phase | 0-50% | **0-60%** |

#### 2. Rewards (trading_env_v2_ultimate.py)

| Paramètre | Avant | Après |
|-----------|-------|-------|
| Trading bonus (BUY/SELL) | +0.03 | **+0.08** |
| HOLD penalty | -0.02 après 30 | **-0.02 immédiat** |
| HOLD penalty extra | -0.05 après 50 | **-0.05 après 10** |
| Diversity bonus | +0.02 | **+0.05** |
| Mode collapse penalty | -0.03 | **-0.05 à -0.10** |

#### 3. Normalisation (trading_env_v2_ultimate.py)

| Avant | Après |
|-------|-------|
| Z-score sur ROW entière (incorrect) | **Z-score PER FEATURE** (correct) |

```python
# AVANT (incorrect)
mean = np.nanmean(base_features)  # Moyenne de tous les 209
std = np.nanstd(base_features)

# APRÈS (correct)
self.feature_means = self.features_df.mean()  # 209 moyennes individuelles
self.feature_stds = self.features_df.std()    # 209 std individuelles
base_features = (base_features - self.feature_means) / self.feature_stds
```

#### 4. Quick Test (tests/quick_test_10k.py)

| Paramètre | Avant | Après |
|-----------|-------|-------|
| `ent_coef` | 0.25 | **0.35** |

---

### FICHIERS MODIFIÉS

1. `environment/trading_env_v2_ultimate.py`
   - Normalisation per-feature
   - Trading bonus +0.08
   - HOLD penalty immédiate
   - Diversity penalties renforcées

2. `training/train_CRITIC_BOOST_LSTM.py`
   - Entropy 0.35 (max)
   - Minimum 0.15
   - High phase 60%

3. `tests/quick_test_10k.py`
   - Entropy 0.35

4. `analysis/agent7_parle.py`
   - Mise à jour des valeurs affichées

---

### FICHIERS SUPPRIMÉS

- `top100_features_agent7.txt` (tous les fichiers top100)

---

### PROCHAINES ÉTAPES

1. **Valider** avec `RUN_QUICK_TEST_10K.bat` (~5 min)
   - Vérifier distribution équilibrée
   - Vérifier trades ouverts > 0

2. **Training complet** avec `LAUNCH_TRAINING_500K.bat` (~6h)
   - From scratch avec nouveaux paramètres
   - Critic devrait apprendre correctement

---

### OBSERVATION SPACE

```
Total: 229 features
- 209 base features (Z-score per feature, clipped [-5, 5])
- 20 RL features (normalized individually)
```

| RL Feature | Range |
|------------|-------|
| Last action (one-hot) | [0, 1] |
| Regret signal | [-1, 1] |
| Position duration | [0, 1] |
| Unrealized PnL | [-5, 5] |
| Market regime (one-hot) | [0, 1] |
| Hours until event | [0, 1] |
| Volatility percentile | [0, 1] |
| Position side | [-1, 0, 1] |
| Trade similarity | [-1, 1] |
| Win rate last 20 | [0, 1] |
| Streak | [-1, 1] |
| Avg PnL last 20 | [-1, 1] |
| Best trade last 20 | [-1, 1] |
| Worst trade last 20 | [-1, 1] |
| Win count last 20 | [0, 1] |
| Loss count last 20 | [0, 1] |

---

*Dernière mise à jour: 2025-12-02 05:15*
