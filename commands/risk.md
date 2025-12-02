---
description: Risk Management FTMO complet - Contraintes, 4R system, position sizing
---

# 🛡️ RISK MANAGEMENT INSTITUTIONNEL - FTMO COMPLIANT

## ⚠️ CONTRAINTES FTMO (CRITIQUES)

### Maximum Drawdown Total : 10%
```python
Calcul: (Peak Balance - Current Balance) / Peak Balance
Vérification: À CHAQUE trade
Action si dépassé: STOP TRADING IMMÉDIAT
Implémentation: GoldTradingEnv.step()

Exemple (Capital 100k):
  Peak Balance: $100,000
  Current Balance: $89,500
  Drawdown: ($100k - $89.5k) / $100k = 10.5% → STOP!
```

### Daily Drawdown : 5%
```python
Calcul: (Balance Start Day - Current Balance) / Balance Start Day
Vérification: À CHAQUE trade
Reset: Minuit UTC
Action si dépassé: STOP TRADING pour la journée

Exemple:
  Balance début jour: $95,000
  Current Balance: $90,250
  Daily DD: ($95k - $90.25k) / $95k = 5% → STOP!
```

### Profit Target (Optionnel)
```
Phase 1 Challenge: 8% profit (non implémenté - focus DD)
Phase 2 Verification: 5% profit (non implémenté)
Note: Focus principal = respect DD, pas profit target
```

---

## 🎯 SYSTÈME 4R (REWARD/RISK)

### Concept
Encourage trades avec bon Risk/Reward ratio

### Définitions
```
1R = 0.25% profit (stop loss typique)
2R = 0.5% profit
3R = 0.75% profit
4R = 1.0% profit (TARGET PRINCIPAL)
```

### Reward Function
```python
reward = pnl_base + bonus_4r + penalite_sortie_precoce - penalite_dd

Composantes:
  pnl_base = Profit/Perte réel du trade (en %)
  
  bonus_4r = +2.0 si profit >= 1.0% (atteint 4R)
    → Encourage patience et holding jusqu'à target
  
  penalite_sortie_precoce = -0.5 si 0 < profit < 0.25% (moins de 1R)
    → Décourage scalping et sorties prématurées
  
  penalite_dd = -(drawdown / max_dd_allowed)^2
    → Apprend à protéger le capital

Exemples:
  Trade parfait 4R (+1.2%):  +1.012 + 2.0 = +3.012 reward ⭐
  Trade moyen 2R (+0.5%):    +0.005 reward
  Trade sortie precoce (+0.2%): +0.002 - 0.5 = -0.498 reward ❌
  Trade perdant (-0.8%):     -0.008 - 0.1 = -0.108 reward
```

### Objectif
- Agents apprennent à viser **4R minimum**
- Évite sorties prématurées (hold jusqu'à 4R)
- **Préfère 1 trade 4R que 4 trades 1R**

### Implémentation
```python
# GoldTradingEnv._calculate_reward() ligne ~200
def _calculate_reward(self, pnl_pct):
    reward = pnl_pct  # Base
    
    if pnl_pct >= 0.01:  # 4R atteint (1%)
        reward += 2.0
    
    if 0 < pnl_pct < 0.0025:  # Sortie avant 1R
        reward -= 0.5
    
    # Pénalité drawdown
    dd_ratio = current_dd / max_dd_allowed
    reward -= dd_ratio ** 2
    
    return np.clip(reward, -10, +10)
```

---

## 💰 POSITION SIZING

### Méthode
**Fixed Fractional** (pourcentage du capital)

### Risk Per Trade
```
Standard: 1-2% du capital
Conservative: 0.5-1% (si drawdown > 5%)
Aggressive: 2-3% (DÉCONSEILLÉ pour FTMO)
```

### Calcul Lots
```python
# Formule
Lots = (Capital * Risk%) / (Stop Loss en pips * Valeur pip)

# Exemple
Capital: $100,000
Risk: 1% = $1,000
Stop Loss: 50 pips
Valeur pip Gold: $1/pip (micro lot)

Lots = $1,000 / (50 pips * $1) = 20 micro lots

# Ajustement si Drawdown
if current_dd > 0.05:  # >5%
    risk_pct *= 0.5  # Réduit risque de moitié
```

### Contraintes
- **Max positions**: 1 seule position à la fois (pas de hedging)
- **Leverage**: 1:30 maximum (FTMO standard)
- **Min lots**: 0.01 (micro lot Gold)
- **Max lots**: Calculé pour ne jamais dépasser 2% risk

---

## 📏 STOPS & TARGETS

### Stop Loss
```python
Méthode: ATR-based OU structure-based

ATR-based:
  SL = Entry ± (ATR14 * 2)
  
  Exemple:
    Entry: $2,050
    ATR14: $10
    SL Long: $2,050 - ($10 * 2) = $2,030
    SL Short: $2,050 + ($10 * 2) = $2,070

Structure-based:
  SL = Below/Above recent swing low/high
  
Contraintes:
  Minimum: 20 pips (spread + slippage)
  Maximum: 100 pips (max 1% capital)
```

### Take Profit
```python
Target Principal: 4R (4x le stop loss)

Calcul:
  Si SL = 20 pips (0.25%)
  TP = 80 pips (1.0%)

Target Secondaire:
  Résistance/Support majeur
  Fibonacci extensions (127%, 161.8%)

Trailing Stop:
  Activation: Après 2R atteint
  Formule: Trail = Max(Entry + 2R, Current High - ATR*1.5)
  Update: Seulement si favorable (jamais recule)
```

### Breakeven
```python
Activation: Après 1.5R atteint

Action:
  Move SL to Entry + spread (0 risk)
  
Exemple:
  Entry: $2,050
  Spread: 2 pips = $2
  Breakeven: Après $2,050 + (1.5 * 0.25%) = $2,057.75
  New SL: $2,050 + $2 = $2,052 (no loss possible)
```

---

## 📊 GESTION POSITION OUVERTE

### Monitoring
- **Fréquence**: À chaque nouvelle bougie H1
- **Vérifications**:
  1. SL pas touché
  2. TP pas atteint
  3. Trailing stop update (si activé)
  4. Daily DD check
  5. Max DD check

### Ajustements Dynamiques

**Trailing Stop:**
```python
if profit >= 2R:
    trail_distance = ATR14 * 1.5
    new_sl = max(current_sl, current_price - trail_distance)
    if new_sl > current_sl:  # Jamais recule
        update_sl(new_sl)
```

**Breakeven:**
```python
if profit >= 1.5R and current_sl < entry:
    update_sl(entry + spread)  # 0 risk
```

**Partial Close:**
```
Statut: NON ACTIVÉ
Raison: RL gère automatiquement (pas de règle fixe)
```

---

## 🚨 CONDITIONS SORTIE

### Sortie Normal
1. **Target atteint** → 4R (1% profit) ✅
2. **Stop touché** → SL (-0.25% loss typique) ❌
3. **Inverse signal** → Agent donne signal opposé fort
4. **Timeout** → Holding > 5 jours ET pas de momentum
5. **Urgence** → Daily DD approche 5% OU Max DD approche 10%

### Cas Spéciaux

**News High Impact (FOMC, NFP, CPI):**
```
Avant: Ferme positions 1h avant si exposé
Après: Attend 15min post-news pour stabilité
Exception: Agent 11 peut trader sur news macro
```

**Weekend Gaps:**
```
Vendredi soir: Ferme positions si holding < 2R
Lundi matin: Attend 2h pour gap comblé
```

**Flash Crash:**
```
Détection: Move > 5% en < 5min
Action: HOLD (pas de panic sell, SL protège)
```

---

## ✅ CHECKLIST PRE-TRADE

Avant CHAQUE trade, vérifier:

- [ ] **Aucune position ouverte** actuellement
- [ ] **Drawdown actuel < 8%** (marge sécurité FTMO)
- [ ] **Pas de news high impact** dans 1h
- [ ] **Spread < 3 pips** (liquidité suffisante)
- [ ] **Au moins 2 agents d'accord** OU Meta confiance > 70%
- [ ] **Volume > SMA20 volume** (confirmation)
- [ ] **ATR dans range normal** (pas extrême)
- [ ] **Heure trading OK** (évite 22h-2h UTC faible liquidité)

---

## 📈 MÉTRIQUES CIBLES (PRODUCTION)

```
ROI Minimum:       > 5% annualisé
Sharpe Ratio:      > 1.0 (excellent > 1.5)
Max Drawdown:      < 10% (FTMO compliance)
Win Rate:          > 45%
Profit Factor:     > 1.5
Calmar Ratio:      > 1.0
Daily Drawdown:    < 5% (FTMO compliance)
Risk/Reward:       > 2.0 (idéal 4.0 avec système 4R)
```

---

**💡 MINDSET**: "Protect capital first, profit second" - Ray Dalio

**🎯 OBJECTIF**: Système 4R garantit que 30-40% win rate suffit pour être profitable (1 win 4R = 4 losses 1R)

**⚠️ CRITIQUE**: JAMAIS violer les limites FTMO - Mieux vaut perdre une opportunité que perdre le compte !
