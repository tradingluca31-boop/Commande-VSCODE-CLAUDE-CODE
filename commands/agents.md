---
description: Détails complets 4 agents + Meta-Agent - Stratégies, features, logique
---

# 🤖 SYSTÈME MULTI-AGENT - DÉTAILS COMPLETS

## Agent 7 : MOMENTUM TRADER (PPO)

### Stratégie
**Approche**: Momentum Trading
**Timeframe**: H1 (1 heure)
**Horizon**: Swing (1-4 jours)

### Logique Trading
1. Détecte phases de momentum fort (RSI + MACD)
2. Entre sur pullbacks dans la tendance
3. Suit le mouvement jusqu'à épuisement
4. Utilise bullish/bearish impulse pour timing

### Features Clés (Top 100)
```
- xauusd_h1_ema_26
- xauusd_h1_ema_12
- xauusd_h1_bullish_impulse
- xauusd_h1_returns_20d
- xauusd_h1_momentum_divergence_5_20
- xauusd_h1_rsi_14
- xauusd_h1_macd_histogram
```

### Conditions Entrée
**BUY:**
- EMA12 > EMA26 (tendance haussière)
- Bullish impulse > seuil
- RSI entre 40-60 (pas suracheté)
- Volume confirme mouvement

**SELL:**
- EMA12 < EMA26 (tendance baissière)
- Bearish impulse > seuil
- RSI entre 40-60 (pas survendu)
- Volume confirme mouvement

### Sortie
- Trailing stop OU inverse de signal

### Performance Attendue
- **Excellent** en trending markets
- **Moyen** en ranging markets
- ROI ~12%, Sharpe ~1.2, DD ~8%

---

## Agent 8 : MEAN REVERSION (SAC)

### Stratégie
**Approche**: Mean Reversion
**Timeframe**: M15 (15 minutes)
**Horizon**: Swing court (1-2 jours)

### Logique Trading
1. Détecte surachat/survente extrêmes
2. Entre sur retour à la moyenne (SMA)
3. Target: retour rapide à SMA20/50
4. Stop serré (mean reversion = high win rate, low RR)

### Features Clés (Top 100)
```
- xauusd_m15_sma_50
- xauusd_m15_sma_200
- xauusd_m15_rsi_14
- xauusd_m15_bb_lower
- xauusd_m15_bb_upper
- xauusd_m15_price_vs_sma50
```

### Conditions Entrée
**BUY:**
- Prix < SMA50 - 2*ATR (oversold)
- RSI < 30 (survente confirmée)
- Touche BB Lower
- Divergence RSI haussière

**SELL:**
- Prix > SMA50 + 2*ATR (overbought)
- RSI > 70 (surachat confirmé)
- Touche BB Upper
- Divergence RSI baissière

### Sortie
- Target = retour SMA20 OU inverse signal

### Performance Attendue
- **Excellent** en ranging markets
- **Faible** en trending markets
- ROI ~8%, Sharpe ~1.0, DD ~9%

---

## Agent 9 : TREND FOLLOWER (TD3)

### Stratégie
**Approche**: Trend Following
**Timeframe**: D1 (Daily)
**Horizon**: Swing long (2-4 jours)

### Logique Trading
1. Identifie tendances long terme (D1)
2. Entre sur breakouts confirmés
3. Holding long (plusieurs jours)
4. Utilise ADX pour force tendance

### Features Clés (Top 100)
```
- xauusd_d1_sma_200
- xauusd_d1_adx_14
- xauusd_d1_momentum_divergence_5_20
- gold_silver_d1_correlation
- xauusd_d1_fib_distance_382
```

### Conditions Entrée
**BUY:**
- Prix > SMA200 (tendance haussière LT)
- ADX > 25 (tendance forte)
- Breakout résistance + volume
- Corrélation Gold/Silver positive

**SELL:**
- Prix < SMA200 (tendance baissière LT)
- ADX > 25 (tendance forte)
- Breakout support + volume
- Divergence baissière confirmée

### Sortie
- ADX < 20 OU casse SMA200

### Performance Attendue
- **Excellent** en forte tendance
- **Faible** en ranging markets
- ROI ~10%, Sharpe ~1.1, DD ~8%

---

## Agent 11 : CORRELATION EXPERT (A2C)

### Stratégie
**Approche**: Macro Correlations + COT
**Timeframe**: D1 + Weekly COT
**Horizon**: Position (3-7 jours)

### Logique Trading
1. Analyse positionnement institutionnel (COT)
2. Détecte divergences COT/Prix (retournements)
3. Utilise macro US (FOMC, NFP) pour timing
4. Trade contre le retail (contrarian)

### Features Clés (Top 100)
```
- cot_gold_noncomm_net_pct
- cot_gold_divergence
- cot_dxy_noncomm_net_pct
- macro_score_emploi
- macro_score_inflation
- macro_score_taux
- retail_gold_contrarian_signal
```

### Conditions Entrée
**BUY:**
- COT institutions nettes long ET en hausse
- Divergence COT haussière (COT up, prix flat/down)
- Macro score emploi/croissance positif
- Retail majoritairement short (contrarian)

**SELL:**
- COT institutions nettes short ET en hausse
- Divergence COT baissière (COT down, prix flat/up)
- Macro score inflation/taux négatif
- Retail majoritairement long (contrarian)

### Sortie
- Inverse divergence COT OU changement macro

### Performance Attendue
- **Excellent** pour retournements
- Signaux rares mais fiables
- ROI ~6%, Sharpe ~0.9, DD ~7%

---

## Meta-Agent : ENSEMBLE ORCHESTRATOR (PPO)

### Fonction
Arbitre intelligent qui sélectionne la meilleure stratégie selon contexte

### Processus Décision

**Étape 1 : Collecte Votes**
- Récupère opinion de chaque agent
- Format: [0=Hold, 1=Buy, 2=Sell]
- Exemple: [1, 0, 1, 2] = Agent7 Buy, Agent8 Hold, Agent9 Buy, Agent11 Sell

**Étape 2 : Analyse Consensus**
- **Consensus fort** (3+ agents d'accord) → Suit avec forte confiance
- **Majorité claire** (2-2) → Analyse contexte marché
- **Désaccord total** → Attend signal clair OU suit agent spécialisé

**Étape 3 : Contexte Marché**
```
Trending Market (ADX > 25, momentum fort):
  → Préfère Agent 9 (TD3 Trend Following) x2 weight

Ranging Market (ADX < 20, BB squeeze):
  → Préfère Agent 8 (SAC Mean Reversion) x2 weight

Momentum Market (Bullish/Bearish impulse fort):
  → Préfère Agent 7 (PPO Momentum) x2 weight

Macro Event (FOMC, NFP dans 24h):
  → Préfère Agent 11 (A2C Correlations) x2 weight
```

**Étape 4 : Décision Finale**
- Weighted average des votes selon contexte
- Si confiance < 60% → Hold
- Output: Action finale (0, 1 ou 2)

### Apprentissage
- **Objectif**: Apprendre pondération optimale dynamiquement
- **Reward**: Basé sur PnL du trade final
- **Optimisation**: Ajuste poids agents selon historique
- **Adaptation**: Si Agent 7 surperforme récemment → + de poids temporairement

### Performance Attendue
- ROI ~15-18% (+20-30% vs meilleur individuel)
- Sharpe ~1.5
- DD ~7%
- **Avantage**: Sélection dynamique du meilleur agent par contexte

---

## Comparaison Agents

| Agent | Algo | Strategy | TF | Best Market | ROI | Sharpe | DD |
|-------|------|----------|----|-----------|----|--------|-----|
| **7** | PPO | Momentum | H1 | Trending | 12% | 1.2 | 8% |
| **8** | SAC | Mean Rev | M15 | Ranging | 8% | 1.0 | 9% |
| **9** | TD3 | Trend | D1 | Strong Trend | 10% | 1.1 | 8% |
| **11** | A2C | COT/Macro | D1 | Reversals | 6% | 0.9 | 7% |
| **Meta** | PPO | Ensemble | Multi | All | 15-18% | 1.5 | 7% |

---

**💡 TIP**: Utilisez `/workflow` pour voir le training complet et `/backtest` pour analyser les résultats.
