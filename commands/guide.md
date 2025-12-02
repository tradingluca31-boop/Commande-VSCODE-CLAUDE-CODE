---
description: Accès expert au guide institutionnel RL Trading Gold - Théories, code, FTMO rules
---

# 📚 GUIDE COMPLET RL TRADING - NIVEAU INSTITUTIONNEL

Accès au guide master : [CLAUDE.md](../CLAUDE.md)

## 🎯 SECTIONS DISPONIBLES

### 1. Théorie RL pour Trading
- State Space Design (features, macro, risk)
- Action Space (discrete vs continuous)
- Reward Function (multi-objectifs, FTMO-aware)
- Hyperparamètres PPO/SAC recommandés

### 2. Stack Technique Pro
- **Stable-Baselines3** (PPO, SAC, A2C, DQN)
- **Gymnasium** (environnements custom)
- **TA-Lib** (indicateurs techniques)
- **QuantStats** (métriques institutionnelles)

### 3. Règles FTMO
- Daily Loss 5%
- Overall DD 10%
- Profit targets
- Trading restrictions

### 4. Backtesting Rigoureux
- Walk-forward analysis
- Monte Carlo simulations
- Out-of-sample testing
- Métriques (Sharpe, Sortino, Calmar, etc.)

### 5. Risk Management
- Position sizing (Kelly Criterion)
- Stop Loss / Take Profit dynamiques
- Correlation risk
- VaR / CVaR

### 6. Code Quality Standards
- Type hints
- Docstrings Google Style
- Tests unitaires
- Logging structuré
- Error handling

### 7. Structure Projet
- Arborescence recommandée
- Séparation data/src/models
- Gestion configs

### 8. Workflow Développement
- Research → Environment → Training → Validation → Production

## 🚀 QUICK START

```python
# 1. Créer environment
from gymnasium import Env
from stable_baselines3 import PPO

class GoldTradingEnv(Env):
    def __init__(self, data, ftmo_rules=True):
        # FTMO constraints integrated
        self.max_daily_loss = 0.05
        self.max_overall_dd = 0.10
        ...

# 2. Entraîner agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 3. Backtest
results = backtest(model, test_data)
sharpe = calculate_sharpe(results['returns'])
```

## 📖 RESSOURCES

### Papers Essentiels
Utilisez `/papers` pour la liste complète

### Books Recommandés
- **"Advances in Financial ML"** - López de Prado
- **"ML for Algorithmic Trading"** - Stefan Jansen
- **"Quantitative Trading"** - Ernest Chan

## ⚡ COMMANDS UTILES

- `/ftmo` - Rappel règles FTMO
- `/papers` - Liste papers académiques
- `#` au début d'un message - Ajouter à Memory
- Shift+Tab - Toggle auto-accept edits

---

**💡 TIP** : Le guide CLAUDE.md est mis à jour régulièrement. Consultez-le fréquemment !
