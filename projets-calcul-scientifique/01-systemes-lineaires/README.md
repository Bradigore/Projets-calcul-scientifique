# 01 – Circuit électrique : résolution d'un système linéaire

**Problème.** Déterminer les 9 courants d'un circuit de 9 résistances alimenté par un générateur de 80 V.

**Méthode.**
- Mise en équations par les lois de Kirchhoff (mailles et nœuds) : système `Ax = b` de taille 9×9.
- Élimination de Gauss avec pivot partiel, codée à la main, puis rétro-substitution.
- Vérification de l'inversibilité (déterminant, rang) et calcul du conditionnement de `A`.

**Résultats.**
- Courant total fourni : 3,92 mA, soit une résistance équivalente de 20,42 kΩ.
- Écart avec `numpy.linalg.solve` : de l'ordre de 10⁻¹⁷ A.
- Conditionnement de `A` ≈ 5,9 × 10⁴ : la matrice est mal conditionnée, ce qui impose le pivot partiel pour limiter les erreurs d'arrondi.

**Lancer.** `python circuit_gauss.py`

Le rapport complet est dans [rapport.pdf](rapport.pdf).
