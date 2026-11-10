# Projets de calcul scientifique – Python

**Semy El Karra** · Élève ingénieur en mathématiques appliquées à la finance, CY Tech

Ce dépôt rassemble quatre projets de méthodes numériques réalisés en Licence 3 de physique
(CY Cergy Paris Université, 2025-2026), ainsi que mon rapport de stage de recherche.
Chaque projet part d'un problème physique, le met en équations, puis le résout numériquement
avec des algorithmes codés à la main et vérifiés contre NumPy.

| Projet | Problème | Méthodes | Lien avec la finance |
| --- | --- | --- | --- |
| [01 – Systèmes linéaires](01-systemes-lineaires/) | Courants dans un circuit de 9 résistances | Pivot de Gauss partiel, rétro-substitution, conditionnement | Résolution de systèmes et stabilité numérique (calibration de modèles) |
| [02 – Modes propres](02-modes-propres/) | Vibrations d'un polymère en anneau | Méthode de la puissance, puissance inverse avec décalage | Valeurs propres de matrices symétriques (ACP, matrices de covariance) |
| [03 – Marche aléatoire](03-marche-aleatoire/) | Du mouvement brownien à l'équation de diffusion | Simulation Monte-Carlo, différences finies (Euler implicite) | Mouvement brownien et équation de la chaleur, à la base du modèle de Black-Scholes |
| [04 – Dynamique moléculaire](04-dynamique-moleculaire/) | Gaz de 64 particules (Lennard-Jones) | Algorithme de Verlet, conditions périodiques, validation statistique | Simulation stochastique et vérification d'une loi de distribution |

Le projet **03** est le plus proche de la finance quantitative : la marche aléatoire converge
vers le mouvement brownien, et l'équation de diffusion résolue ici par différences finies
est celle à laquelle se ramène l'EDP de Black-Scholes.

Le dossier [rapport-stage](rapport-stage/) contient mon rapport de stage de recherche (4 semaines,
Laboratoire de physique théorique) sur les potentiels de Liénard-Wiechert : dérivation analytique
complète par fonctions de Green et distributions de Dirac.

## Lancer les projets

```bash
pip install -r requirements.txt
cd 03-marche-aleatoire
python marche_aleatoire_diffusion.py
```

Outils : Python 3, NumPy, Matplotlib, LaTeX (rapports).
