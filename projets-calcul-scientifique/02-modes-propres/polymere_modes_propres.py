"""
Projet II – Étude d'un polymère homogène : modes propres de vibration
Semy El Karra – Licence 3 Physique, CY Cergy Paris Université

Le polymère est modélisé par une chaîne fermée (boucle) de n monomères de masse m,
reliés par des ressorts de raideur K. Les équations de Newton se mettent sous la forme
A x = λ x, avec λ = ω². La matrice A est symétrique réelle (théorème spectral).

Algorithmes utilisés (détaillés dans le rapport) :
  - Élimination de Gauss avec pivot partiel + rétro-substitution
  - Méthode de la puissance : valeur propre dominante (mode de plus haute fréquence)
  - Méthode de la puissance inverse avec décalage : modes de basse fréquence

Le programme détermine :
  - le premier mode excité (le mode nul correspond à une translation de l'ensemble),
  - le mode de plus haute fréquence,
  - le nombre de modes propres,
puis vérifie les résultats avec numpy.linalg.eigh.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# DONNÉES PHYSIQUES
n = 6   # Nombre de monomères (ex : hexagone)
K = 2   # Raideur des ressorts (N/m)
m = 1   # Masse d'un monomère (unités réduites)


# ------------------------------------------------------------------------------------
# Élimination de Gauss avec pivot partiel
def gauss(A, b, precision=1e-12):
    """Triangularise le système A y = b (utilisée dans la puissance inverse pour (A - zI) y = x).

    1. Recherche du pivot maximal dans la colonne courante pour limiter les erreurs numériques.
    2. Échange de lignes afin de placer ce pivot sur la diagonale.
    3. Élimination des coefficients situés en dessous du pivot.
    4. Répétition pour chaque colonne.
    On travaille sur des copies en réels pour ne pas modifier les matrices initiales.
    """
    A_copy = A.copy().astype(float)
    b_copy = b.copy().astype(float)
    n_lig = len(b_copy)

    for k in range(n_lig):
        # Étape 1 : position du pivot maximal dans la colonne k (à partir de la ligne k)
        p = np.argmax(abs(A_copy[k:, k])) + k

        # Étape 2 : vérification du pivot
        if abs(A_copy[p, k]) < precision:
            return None, None  # Matrice (quasi) singulière

        # Étape 3 : permutation des lignes si le pivot n'est pas déjà sur la diagonale
        if p != k:
            A_copy[[k, p]] = A_copy[[p, k]]
            b_copy[k], b_copy[p] = b_copy[p], b_copy[k]

        # Étape 4 : élimination des coefficients sous le pivot
        for i in range(k + 1, n_lig):
            h = A_copy[i, k] / A_copy[k, k]
            A_copy[i, k:] = A_copy[i, k:] - h * A_copy[k, k:]
            b_copy[i] = b_copy[i] - h * b_copy[k]

    return A_copy, b_copy


# ------------------------------------------------------------------------------------
# Rétro-substitution
def retrosub(A, b):
    """Résout un système triangulaire supérieur, de la dernière ligne vers la première."""
    n_lig = len(b)
    X = np.zeros(n_lig)
    for i in range(n_lig - 1, -1, -1):
        s = 0  # Somme des termes déjà connus
        for j in range(i + 1, n_lig):
            s += A[i, j] * X[j]
        X[i] = (b[i] - s) / A[i, i]
    return X


# ------------------------------------------------------------------------------------
# Méthode de la puissance
def methode_puissance(A, precision=1e-12, iterations=1000):
    """Détermine la valeur propre dominante et son vecteur propre.
    Critère de convergence : |q(k) - q(k-1)| < precision (critère de Cauchy)."""
    n_lig = A.shape[0]
    # Vecteur initial aléatoire normalisé : ||x|| = 1
    x = np.random.rand(n_lig)
    x = x / math.sqrt(sum(x[i]**2 for i in range(n_lig)))
    q_old = 0

    for _ in range(iterations):
        # y = A x(k-1) : produit matrice-vecteur
        y = np.zeros(n_lig)
        for i in range(n_lig):
            for j in range(n_lig):
                y[i] += A[i, j] * x[j]

        # Normalisation : x(k) = y / ||y||
        x = y / math.sqrt(sum(y[i]**2 for i in range(n_lig)))

        # Quotient de Rayleigh : q(k) = xᵀ (A x)
        Ax = np.zeros(n_lig)
        for i in range(n_lig):
            for j in range(n_lig):
                Ax[i] += A[i, j] * x[j]
        q = sum(x[i] * Ax[i] for i in range(n_lig))

        if abs(q - q_old) < precision:
            break  # Convergence atteinte
        q_old = q

    return q, x  # q = valeur propre dominante, x = vecteur propre associé


# ------------------------------------------------------------------------------------
# Méthode de la puissance inverse avec décalage
def puissance_inverse(A, z=0.1, precision=1e-12, iterations=1000):
    """Méthode de la puissance inverse avec décalage.
    On résout (A - z(k) I) y = x(k-1) avec Gauss + rétro-substitution.
    Le décalage est fixe pendant les 5 premières itérations, puis z = q(k-1)."""
    n_lig = A.shape[0]
    I = np.eye(n_lig)

    # Vecteur initial aléatoire normalisé
    x = np.random.rand(n_lig)
    x = x / math.sqrt(sum(x[i]**2 for i in range(n_lig)))
    q = z

    for k in range(1, iterations + 1):
        if k > 5:
            z = q  # Décalage adaptatif

        # Résolution linéaire (A - zI) y = x
        A_tri, b_prime = gauss(A - z * I, x)
        if A_tri is None:
            # z est exactement une valeur propre : on décale légèrement et on réessaie
            z = z + 0.1
            A_tri, b_prime = gauss(A - z * I, x)
            if A_tri is None:
                return q, x
        y = retrosub(A_tri, b_prime)

        # Normalisation
        norme = math.sqrt(sum(y[i]**2 for i in range(n_lig)))
        x = y / norme

        # Quotient de Rayleigh
        Ax = np.zeros(n_lig)
        for i in range(n_lig):
            for j in range(n_lig):
                Ax[i] += A[i, j] * x[j]
        q_new = sum(x[i] * Ax[i] for i in range(n_lig))

        # Convergence (critère de Cauchy)
        if abs(q_new - q) < precision:
            q = q_new
            break
        q = q_new

    return q, x


# ------------------------------------------------------------------------------------
def matrice(n, K_Mat, m=1.0):
    """Construit la matrice dynamique A = K / m de la chaîne fermée.

    Chaque monomère i est relié à son voisin précédent (i-1) et à son voisin suivant (i+1).
    La chaîne étant une boucle, on utilise le modulo :
    - (i-1) % n donne n-1 quand i = 0,
    - (i+1) % n donne 0 quand i = n-1.
    """
    A = np.zeros((n, n))
    for i in range(n):
        i_prec = (i - 1) % n
        i_suiv = (i + 1) % n
        # Élément diagonal : somme des raideurs des deux ressorts qui tirent sur la masse i
        A[i, i] = (K_Mat[i_prec] + K_Mat[i]) / m
        # Liaisons avec les voisins
        A[i, i_prec] = -K_Mat[i_prec] / m
        A[i, i_suiv] = -K_Mat[i] / m
    return A


# ------------------------------------------------------------------------------------
def trouver_tous_modes_non_nuls(A, precision=1e-8):
    """Balaye plusieurs décalages z avec la puissance inverse pour trouver
    les valeurs propres distinctes non nulles (le mode nul est une translation)."""
    modes_trouves = []
    # Décalages variés pour couvrir tout le spectre
    z_values = [0.3, 0.8, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5]

    for z in z_values:
        val, vec = puissance_inverse(A, z=z, precision=precision)
        if abs(val) > precision:
            est_nouveau = all(abs(v - val) > precision * 100 for v, _ in modes_trouves)
            if est_nouveau:
                modes_trouves.append((val, vec))

    modes_trouves.sort(key=lambda x: x[0])  # Ordre croissant
    return modes_trouves


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)  # Résultats reproductibles

    # Mise en équation du système matriciel
    K_Mat = [K] * n
    A = matrice(n, K_Mat, m)
    print("Matrice A :")
    print(A)

    # Modes non nuls par la puissance inverse
    modes = trouver_tous_modes_non_nuls(A)
    print("\nValeurs propres distinctes non nulles (puissance inverse) :")
    for val, _ in modes:
        print(f"  λ = {val:.7f}   ω = {np.sqrt(val):.7f}")

    # Premier mode excité = plus petite valeur propre non nulle
    lam1, v1 = modes[0]
    print(f"\nPremier mode excité : λ = {lam1:.7f}, ω = {np.sqrt(lam1):.7f}")
    print("Vecteur propre :", np.round(v1, 4))

    # Mode de plus haute fréquence : méthode de la puissance directe
    lam_max, v_max = methode_puissance(A)
    print(f"\nMode de plus haute fréquence : λ_max = {lam_max:.7f}, ω_max = {np.sqrt(lam_max):.7f}")
    print("Vecteur propre :", np.round(v_max, 4))

    # Vérification par diagonalisation exacte (numpy)
    vals, vecs = np.linalg.eigh(A)
    print("\nVérification numpy.linalg.eigh :")
    print("  Valeurs propres :", np.round(vals, 6))
    print("  Pulsations      :", np.round(np.sqrt(np.clip(vals, 0, None)), 6))
    print(f"  Nombre de modes propres : {n} (dont 1 mode nul de translation, et des modes dégénérés)")

    # Figure : déplacement des monomères pour le premier mode excité et le mode le plus rapide
    indices = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].plot(indices, v1, "o-")
    axes[0].set_title(f"Premier mode excité (ω = {np.sqrt(lam1):.4f})")
    axes[1].plot(indices, v_max, "o-", color="tab:red")
    axes[1].set_title(f"Mode de plus haute fréquence (ω = {np.sqrt(lam_max):.4f})")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Indice du monomère i")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Déplacement (u.a.)")
    plt.tight_layout()
    plt.savefig("modes_propres.png", dpi=150)
    plt.show()
