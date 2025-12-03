# 📘 MASTER WORKFLOW GUIDE - Système Updates/ pour Agents RL

> **À COPIER-COLLER** pour chaque nouvel agent ou amélioration

---

## 🎯 Concept

Au lieu de commiter directement sur `main`, on crée un **dossier daté** dans `updates/` pour :
- Tester les changements isolément
- Documenter proprement
- Garder un historique clair
- Rollback facile si problème

---

## 📂 Structure Standard

```
GoldRL/
  └── updates/
      ├── MASTER_WORKFLOW_GUIDE.md  ← CE FICHIER
      │
      ├── YYYY-MM-DD-description-courte/  ← Format OBLIGATOIRE
      │   ├── DESCRIPTION.md             ← OBLIGATOIRE (template ci-dessous)
      │   ├── RESULTS.txt                ← Résultats training/tests
      │   ├── fichiers_modifiés.py       ← Copie des fichiers changés
      │   ├── BENCHMARK.csv              ← Métriques de performance
      │   └── screenshots/               ← (optionnel) Images
      │
      └── DEPLOYED-YYYY-MM-DD-xxx/       ← Updates déployées (archivées)
```

---

## 🔧 Workflow Étape par Étape

### 1️⃣ Créer un Nouveau Update

```bash
# Aller dans le projet
cd C:\Users\lbye3\Desktop\GoldRL

# Créer le dossier (REMPLACER LA DATE ET DESCRIPTION)
mkdir updates\2025-12-03-ma-nouvelle-feature

# Entrer dans le dossier
cd updates\2025-12-03-ma-nouvelle-feature
```

**Naming Convention** :
- `YYYY-MM-DD-description-courte`
- Exemples :
  - `2025-12-03-dataset-2008-training-500k`
  - `2025-12-04-agent8-mean-reversion`
  - `2025-12-05-meta-agent-integration`
  - `2025-12-10-fix-checkpoint-loading-bug`

---

### 2️⃣ Créer le Fichier DESCRIPTION.md

**Template à copier-coller** :

```markdown
# Update: [TITRE DE L'UPDATE]

## 📅 Informations
- **Date**: YYYY-MM-DD
- **Agent**: Agent X
- **Status**: ⏳ IN PROGRESS | ✅ COMPLETED | ❌ FAILED | 🚀 DEPLOYED
- **Next**: [Prochaines étapes]

---

## 🔄 Changements Appliqués

### 1. [Nom du changement]
```python
# AVANT
ancien_code = valeur

# APRÈS
nouveau_code = nouvelle_valeur
```

**Impact**:
- [Impact 1]
- [Impact 2]

---

### 2. [Autre changement]
[Description...]

---

## 📊 Résultats

### Performance
- **Métrique 1**: XX%
- **Métrique 2**: XX
- **Métrique 3**: X.XX

### Comparaison Avant/Après
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Win Rate | XX%   | YY%   | +ZZ%         |
| ROI      | XX%   | YY%   | +ZZ%         |

---

## ⏱️ Durée

**Training/Tests**:
- Durée: Xh
- Steps: XXX,XXX
- Dataset: [description]

---

## 🎯 Prochaines Étapes

### Phase 1: [Description]
- [ ] Todo 1
- [ ] Todo 2

### Phase 2: [Description]
- [ ] Todo 3

---

## 🔍 Analyse

### Points Forts
✅ [Point fort 1]
✅ [Point fort 2]

### Points à Améliorer
⚠️ [Point à améliorer 1]
⚠️ [Point à améliorer 2]

### Recommandations
1. [Recommandation 1]
2. [Recommandation 2]

---

## 📁 Fichiers Modifiés

- `chemin/fichier1.py` - [Description changement]
- `chemin/fichier2.py` - [Description changement]

---

## 🐛 Bugs/Issues

- [ ] Bug 1: [Description]
- [x] Bug 2: [RÉSOLU] [Description]

---

## 📝 Notes

[Autres notes importantes...]

---

**Créé par**: [Votre nom / Claude Code]
**Version**: [Version de l'agent]
**Commit**: [SHA du commit Git si applicable]
```

---

### 3️⃣ Copier les Fichiers Modifiés

```bash
# Copier les fichiers changés dans le dossier update
copy "C:\chemin\vers\fichier_modifié.py" "updates\2025-12-03-xxx\fichier_modifié.py"

# Exemple concret
copy "config.py" "updates\2025-12-03-dataset-2008\config.py"
copy "training\train.py" "updates\2025-12-03-dataset-2008\train.py"
```

---

### 4️⃣ Tester l'Update

```bash
# Méthode 1: Utiliser directement les fichiers du dossier updates/
python updates\2025-12-03-xxx\train.py

# Méthode 2: Copier temporairement dans le projet
copy "updates\2025-12-03-xxx\config.py" "config.py.TEST"
# Tester...
# Restaurer
del config.py.TEST
```

---

### 5️⃣ Documenter les Résultats

**Créer RESULTS.txt** :

```txt
================================================================================
RESULTS - Update 2025-12-03
================================================================================

TRAINING COMPLETED
  Duration: 8h30min
  Steps: 500,000
  Final Loss: 0.025

PERFORMANCE METRICS
  Win Rate: 65.78%
  ROI: 9.30%
  Sharpe: 1.35
  Max DD: 4.28%
  Profit Factor: 1.36

CHECKPOINTS SAVED
  - checkpoint_50000.zip
  - checkpoint_100000.zip
  - ...
  - checkpoint_500000.zip

BEST CHECKPOINT
  Step: 200,000
  Score: 7.99/10
  Win Rate: 68.97%

NEXT STEPS
  1. Continue to 1M steps
  2. Compare 200K vs 500K
  3. Backtest on 2021-2024

================================================================================
```

---

### 6️⃣ Décision : Déployer ou Non ?

#### ✅ SI RÉSULTATS BONS : Déployer

```bash
# Copier les fichiers dans le projet principal
copy "updates\2025-12-03-xxx\config.py" "config.py"
copy "updates\2025-12-03-xxx\train.py" "training\train.py"

# Renommer le dossier pour indiquer qu'il est déployé
move "updates\2025-12-03-xxx" "updates\DEPLOYED-2025-12-03-xxx"

# Mettre à jour le statut dans DESCRIPTION.md
# Status: 🚀 DEPLOYED
```

#### ❌ SI RÉSULTATS MAUVAIS : Archiver

```bash
# Renommer pour indiquer l'échec
move "updates\2025-12-03-xxx" "updates\FAILED-2025-12-03-xxx"

# Mettre à jour DESCRIPTION.md avec les raisons de l'échec
# Status: ❌ FAILED
```

---

## 📋 Checklist Update Complet

- [ ] Créer dossier `updates/YYYY-MM-DD-description/`
- [ ] Créer `DESCRIPTION.md` (utiliser template)
- [ ] Copier fichiers modifiés dans le dossier
- [ ] Tester les changements
- [ ] Documenter résultats dans `RESULTS.txt`
- [ ] Prendre décision : Déployer / Archiver / Continuer
- [ ] Si déployé : Renommer en `DEPLOYED-xxx`
- [ ] Si échec : Renommer en `FAILED-xxx`

---

## 🎯 Exemples Concrets

### Exemple 1 : Nouveau Dataset

```
updates/
  └── 2025-12-03-dataset-2008-training-500k/
      ├── DESCRIPTION.md          ← Changements dataset
      ├── config.py               ← TRAIN_START_DATE modifié
      ├── RESULTS.txt             ← Métriques training
      └── BENCHMARK_500K.csv      ← Performance checkpoints
```

### Exemple 2 : Nouveau Callback

```
updates/
  └── 2025-12-04-adaptive-entropy-callback/
      ├── DESCRIPTION.md
      ├── callbacks.py            ← Nouveau callback
      ├── train.py                ← Intégration callback
      ├── RESULTS.txt
      └── comparison_avant_apres.png
```

### Exemple 3 : Fix Bug

```
updates/
  └── 2025-12-05-fix-checkpoint-loading-bug/
      ├── DESCRIPTION.md
      ├── utils.py                ← Bug fix
      ├── test_fix.py             ← Test unitaire
      └── RESULTS.txt             ← Confirmation fix
```

---

## 🚀 Commandes Rapides

```bash
# Créer un nouvel update
cd C:\Users\lbye3\Desktop\GoldRL
mkdir updates\$(date +%Y-%m-%d)-ma-feature

# Lister tous les updates
dir updates

# Voir les updates déployés
dir updates\DEPLOYED-*

# Voir les updates en échec
dir updates\FAILED-*

# Restaurer un ancien update
copy "updates\DEPLOYED-2025-12-03-xxx\config.py" "config.py"
```

---

## 💡 Bonnes Pratiques

### ✅ À FAIRE

1. **Toujours dater** : `YYYY-MM-DD-` au début du nom
2. **Nom descriptif** : `dataset-2008` pas `update1`
3. **DESCRIPTION.md complet** : Utiliser le template
4. **Tester avant déployer** : Jamais déployer sans tests
5. **Documenter résultats** : Créer `RESULTS.txt`
6. **Renommer après déploiement** : `DEPLOYED-` ou `FAILED-`

### ❌ À ÉVITER

1. ❌ Noms génériques : `test`, `new`, `update`
2. ❌ Oublier la date : Impossible de trier chronologiquement
3. ❌ Pas de documentation : Vous oublierez dans 1 mois
4. ❌ Déployer sans tester : Risque de casser le projet
5. ❌ Garder updates déployés dans racine : Archiver !

---

## 🔄 Cycle de Vie d'un Update

```
1. Création
   updates/2025-12-03-ma-feature/  (Status: ⏳ IN PROGRESS)

2. Tests
   → Tests OK ✅
   → Tests KO ❌

3a. SI OK : Déploiement
   updates/DEPLOYED-2025-12-03-ma-feature/  (Status: 🚀 DEPLOYED)

3b. SI KO : Archive
   updates/FAILED-2025-12-03-ma-feature/  (Status: ❌ FAILED)
```

---

## 📞 Aide Rapide

**Q: Combien d'updates puis-je avoir ?**
A: Autant que vous voulez ! Mais archivez (DEPLOYED/FAILED) régulièrement.

**Q: Puis-je avoir plusieurs updates actifs ?**
A: Oui, mais max 2-3 pour ne pas vous perdre.

**Q: Que faire si j'oublie de documenter ?**
A: Mieux vaut tard que jamais ! Créez `DESCRIPTION.md` même après coup.

**Q: Puis-je supprimer les updates DEPLOYED ?**
A: OUI, après 1-2 mois si tout fonctionne bien. Mais gardez les FAILED (pour apprendre).

**Q: Comment partager avec d'autres ?**
A: Zipper le dossier `updates/YYYY-MM-DD-xxx/` et partager.

---

## 🎓 Résumé en 3 Lignes

1. **Créer** : `updates/YYYY-MM-DD-description/` + `DESCRIPTION.md`
2. **Tester** : Copier fichiers, tester, documenter `RESULTS.txt`
3. **Décider** : Déployer (`DEPLOYED-`) ou Archiver (`FAILED-`)

---

**Ce guide est votre RÉFÉRENCE permanente. Consultez-le à chaque nouveau update !**

🤖 Generated with Claude Code