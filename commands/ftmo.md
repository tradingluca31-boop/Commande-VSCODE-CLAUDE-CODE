---
description: Rappel des règles FTMO critiques pour le trading
---

# ⚠️ RAPPEL RÈGLES FTMO - PRIORITÉ ABSOLUE

## 🚨 RÈGLES NON-NÉGOCIABLES

### 1. Maximum Daily Loss : 5%
- Calculé sur balance de **début de journée**
- Si atteint → **STOP TRADING** immédiatement
- Pas de "revenge trading"
- Monitoring en temps réel obligatoire

### 2. Maximum Overall Drawdown : 10%
- Calculé depuis **balance initiale**
- Si atteint → **COMPTE FERMÉ** définitivement
- Risk management ultra-strict nécessaire
- Position sizing conservateur (<2% par trade)

### 3. Profit Target
- **Phase 1** : +10% pour passer en Phase 2
- **Phase 2** : +5% pour obtenir compte financé
- **Minimum 4 jours** de trading actif

### 4. Minimum Trading Days
- **Phase 1** : 4 jours minimum
- **Phase 2** : 4 jours minimum
- Au moins **1 position** ouverte par jour comptabilisé

### 5. Trading Interdit
- **Weekends** (samedi/dimanche)
- **Jours fériés** majeurs (Noël, Nouvel An, etc.)
- **2 minutes** avant/après news à fort impact :
  - NFP (Non-Farm Payrolls)
  - FOMC (Fed meetings)
  - CPI (Inflation data)
  - GDP (PIB)

## 📊 Calculs Critiques

### Daily Loss Example (compte $100,000)
```
Balance début journée : $100,000
Max daily loss : $5,000 (5%)
Si balance atteint : $95,000 → STOP
```

### Overall Drawdown Example
```
Balance initiale : $100,000
Max overall DD : $10,000 (10%)
Balance minimale acceptée : $90,000
Si atteint → COMPTE FERMÉ
```

## 🛡️ INTÉGRATION RL

### Reward Function Penalty
```python
# Pénalité MASSIVE si violation
if daily_loss_pct > 0.05:
    reward -= 1000  # Kill l'épisode
    
if overall_dd_pct > 0.10:
    reward -= 10000  # Fin de l'entraînement
```

### State Monitoring
```python
state['daily_loss_pct'] = ...
state['overall_dd_pct'] = ...
state['trading_days_count'] = ...
state['is_news_period'] = ...  # 2min before/after
```

## ⚡ CHECKLIST PRE-TRADE

Avant CHAQUE trade, vérifier :
- [ ] Daily loss < 4% (marge de sécurité)
- [ ] Overall DD < 8% (marge de sécurité)
- [ ] Pas de news à fort impact dans 2min
- [ ] Pas weekend/férié
- [ ] Stop loss défini et respecté

**🎯 OBJECTIF** : Survivre d'abord, profiter ensuite.
**💡 MINDSET** : "Don't lose money" - Warren Buffett Rule #1
