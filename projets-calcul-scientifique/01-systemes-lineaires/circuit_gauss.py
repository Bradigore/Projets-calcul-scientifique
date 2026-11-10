"""
Projet I – Circuit électrique : résolution d'un système linéaire Ax = b
Semy El Karra – Licence 3 Physique, CY Cergy Paris Université

Ce code résout le système d'équations issu des lois de Kirchhoff (mailles et nœuds)
pour un circuit de 9 résistances alimenté par un générateur idéal.
Méthode : élimination de Gauss avec pivot partiel, puis rétro-substitution.
La solution est vérifiée avec numpy.linalg.solve.
"""

import numpy as np

# Données
Vg = 80  # Tension du générateur (V)
# Tableau des résistances : pratique en cas de modification des valeurs
R = np.array([2.5, 10, 5.1, 8.2, 7.9, 3.6, 3.3, 3, 7.1]) * 1000.0  # Résistances en ohms

# Matrice A 9x9 (important : elle doit être déclarée en float)
A = np.array([
    # Mailles (5 équations)
    [R[0], 0, -R[2], 0, R[4], 0, 0, -R[7], 0],   # Maille 1
    [0, R[1], 0, 0, -R[4], 0, R[6], 0, -R[8]],   # Maille 2
    [0, R[1], 0, R[3], 0, -R[5], 0, -R[7], 0],   # Maille 3
    [R[0], R[1], 0, 0, 0, 0, R[6], 0, 0],        # Maille 4
    [0, 0, 0, 0, 0, R[5], R[6], 0, R[8]],        # Maille 5
    # Nœuds (4 équations)
    [0, 0, 1, -1, 0, 0, 0, -1, 0],   # Nœud A
    [1, -1, 0, 1, -1, 0, 0, 0, 0],   # Nœud B
    [0, 1, 0, 0, 0, 1, -1, 0, 0],    # Nœud C
    [0, 0, 0, 0, 1, -1, 0, 1, -1]    # Nœud D
])

# Second membre b
b = np.array([0, 0, 0, Vg, 0, 0, 0, 0, 0])

precision = 1e-4


def gauss(A, b, verbose=False):
    """Triangularise le système Ax = b par élimination de Gauss avec pivot partiel.

    1. Recherche du pivot maximal dans la colonne courante pour limiter les erreurs numériques.
    2. Échange de lignes afin de placer ce pivot sur la diagonale.
    3. Élimination des coefficients situés en dessous du pivot.
    4. Répétition pour chaque colonne.

    On travaille sur des copies pour garder A et b intacts (comparaison avec numpy ensuite).
    Déclarer les matrices en réels est important : en entiers, les divisions
    peuvent fausser les calculs et faire apparaître un pivot nul.
    """
    A_copy = A.copy().astype(float)
    b_copy = b.copy().astype(float)
    n = len(b_copy)  # Nombre d'équations

    # On parcourt chaque colonne k de la matrice
    for k in range(n):
        p = k  # Indice du pivot : on suppose d'abord qu'il est sur la diagonale
        maxi = abs(A_copy[k, k])
        # Étape 1 : on cherche, dans les lignes en dessous, une valeur plus grande dans la même colonne
        for i in range(k + 1, n):
            if abs(A_copy[i, k]) > maxi:
                maxi = abs(A_copy[i, k])
                p = i  # On enregistre la position de la ligne

        # Étape 2 : vérification du pivot
        if maxi < precision:
            print("Pivot nul : la matrice n'est pas inversible")
            return None, None

        # Étape 3 : permutation des lignes si le pivot n'est pas déjà sur la diagonale
        if p != k:
            A_copy[[k, p]] = A_copy[[p, k]]
            # On échange aussi les valeurs correspondantes de b (on ne travaille pas sur la matrice augmentée)
            b_copy[k], b_copy[p] = b_copy[p], b_copy[k]
            if verbose:
                print(f"\nPermutation L{k+1} <-> L{p+1}")
                print(A_copy)

        # Étape 4 : élimination des coefficients sous le pivot
        for i in range(k + 1, n):
            h = A_copy[i, k] / A_copy[k, k]
            # On soustrait h fois la ligne k à la ligne i
            for j in range(k, n):
                A_copy[i, j] = A_copy[i, j] - h * A_copy[k, j]
            # Même opération sur b
            b_copy[i] = b_copy[i] - h * b_copy[k]

    # On retourne la matrice triangulaire supérieure et le vecteur b modifié
    return A_copy, b_copy


def retrosub(A, b):
    """Résout un système triangulaire supérieur par rétro-substitution."""
    n = len(b)
    X = np.zeros(n)  # Vecteur des inconnues
    # On parcourt les lignes de A du bas vers le haut
    for i in range(n - 1, -1, -1):
        s = 0  # Somme des termes déjà connus
        for j in range(i + 1, n):
            s = s + A[i, j] * X[j]
        # Formule de substitution arrière : x[i] = (b[i] - s) / coefficient diagonal
        X[i] = (b[i] - s) / A[i, i]
    return X


if __name__ == "__main__":
    # Vérifications préalables (non indispensables) : inversibilité et conditionnement de A
    print("det(A)  =", np.linalg.det(A))
    print("rang(A) =", np.linalg.matrix_rank(A))
    print(f"cond(A) = {np.linalg.cond(A):.4e}")

    A_tri, b_prime = gauss(A, b)

    # Si la triangularisation a réussi, la rétro-substitution donne les intensités
    if A_tri is not None:
        X = retrosub(A_tri, b_prime)
        print("\nCOURANTS (mA) :")
        for i in range(9):
            print(f"I{i+1} = {X[i]*1000:.4f} mA")

        print(f"\nCourant total : I1 + I3 = {(X[0] + X[2])*1000:.4f} mA")
        print(f"Résistance équivalente : {Vg/(X[0] + X[2])/1000:.4f} kOhm")

        # Comparaison avec la solution de numpy
        X_numpy = np.linalg.solve(A.astype(float), b.astype(float))
        print(f"\nÉcart maximal avec numpy.linalg.solve : {np.max(np.abs(X - X_numpy)):.2e} A")
