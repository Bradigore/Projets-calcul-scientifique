"""
Projet III – Marche aléatoire et équation de diffusion
Semy El Karra – Licence 3 Physique, CY Cergy Paris Université

Question : comment un mouvement aléatoire microscopique donne-t-il naissance
à un comportement diffusif macroscopique ?

1. Simulation Monte-Carlo de marches aléatoires sur un réseau carré 2D
   et estimation du coefficient de diffusion D par régression linéaire de <r²> = 4 D n.
2. Résolution numérique de l'équation de diffusion 1D par un schéma d'Euler implicite
   (système tridiagonal résolu par Gauss avec pivot partiel + rétro-substitution).
3. Comparaison avec la solution analytique gaussienne P(x,t) = exp(-x²/4Dt) / √(4πDt).

Fonctions NumPy utilisées :
np.zeros, np.copy, np.argmax, np.argmin, np.abs, np.linspace, np.sqrt, np.exp,
np.pi, np.sum, np.mean, np.array, np.polyfit
"""

import numpy as np
import matplotlib.pyplot as plt
import random


# ------------------------------------------------------------------------------------
# Méthode de Gauss avec pivot partiel
def gauss(A, b, precision=1e-10):
    """Triangularise le système A x = b (pivot partiel). On travaille sur des copies."""
    A_copy = A.copy().astype(float)
    b_copy = b.copy().astype(float)
    n = len(b_copy)  # Nombre d'équations

    for k in range(n - 1):  # On traite les n-1 premières colonnes
        # Étape 1 : recherche du pivot maximal dans la colonne k
        # np.argmax donne la position relative, on ajoute k pour l'indice absolu
        p = np.argmax(abs(A_copy[k:, k])) + k

        # Étape 2 : vérification du pivot (matrice singulière si trop proche de zéro)
        if abs(A_copy[p, k]) < precision:
            print("Pivot nul")
            return None, None

        # Étape 3 : permutation des lignes si le pivot n'est pas déjà sur la diagonale
        if p != k:
            A_copy[[k, p]] = A_copy[[p, k]]
            b_copy[k], b_copy[p] = b_copy[p], b_copy[k]

        # Étape 4 : élimination des coefficients sous le pivot
        for i in range(k + 1, n):
            h = A_copy[i, k] / A_copy[k, k]  # Facteur d'élimination
            # Slicing NumPy pour limiter la complexité
            A_copy[i, k:] = A_copy[i, k:] - h * A_copy[k, k:]
            b_copy[i] = b_copy[i] - h * b_copy[k]

    return A_copy, b_copy


# Rétro-substitution
def retrosub(A, b):
    """Résout un système triangulaire supérieur, du bas vers le haut."""
    n = len(b)
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0  # Somme des termes déjà connus
        for j in range(i + 1, n):
            s = s + A[i, j] * X[j]
        # Formule de substitution arrière : x[i] = (b[i] - s) / coefficient diagonal
        X[i] = (b[i] - s) / A[i, i]
    return X


# ------------------------------------------------------------------------------------
def construire_matrice_diffusion(N, D, dt, dx):
    """Matrice tridiagonale du schéma d'Euler implicite : A P^(n+1) = P^n."""
    # r = D Δt / Δx² apparaît dans la discrétisation
    r = D * dt / dx**2

    A = np.zeros((N, N))
    # Équations des points intérieurs (i = 1 à N-2)
    for i in range(1, N - 1):
        A[i, i - 1] = -r       # Voisin gauche P_{i-1}
        A[i, i] = 1 + 2 * r    # Point central P_i
        A[i, i + 1] = -r       # Voisin droit P_{i+1}

    # Conditions aux limites : P(0,t) = 0 et P(L,t) = 0
    A[0, 0] = 1
    A[-1, -1] = 1
    return A


def euler_diffusion(P0, D, dt, dx, nb_pas):
    """Fait évoluer P par le schéma d'Euler implicite et renvoie toute l'évolution."""
    N = len(P0)  # Nombre de points de grille
    A = construire_matrice_diffusion(N, D, dt, dx)
    P = P0.copy()
    evolution = [P.copy()]  # On stocke déjà la condition initiale P(x,0)

    # Boucle temporelle : à chaque pas, on résout A P^(n+1) = P^n
    for _ in range(nb_pas):
        A_tri, b_tri = gauss(A.copy(), P.copy())  # Élimination de Gauss avec pivot partiel
        P = retrosub(A_tri, b_tri)                # Substitution arrière -> P^(n+1)
        evolution.append(P.copy())

    return np.array(evolution)


def solution_analytique_diffusion(x, t, D, dx, x0=0):
    """Solution fondamentale de l'équation de diffusion (condition initiale de Dirac en x0)."""
    if t == 0:
        # Approximation numérique de la distribution de Dirac :
        # probabilité concentrée au point de grille le plus proche de x0, d'intégrale 1
        P = np.zeros(len(x))
        i0 = np.argmin(np.abs(x - x0))
        P[i0] = 1 / dx
        return P

    # Cas général t > 0 : P(x,t) = 1/√(4πDt) · exp(-(x-x0)²/(4Dt)), de variance σ² = 2Dt
    return np.exp(-(x - x0)**2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)


def graphique_euler(D):
    """Compare la solution numérique (Euler implicite) et la solution analytique."""
    # 1. Domaine spatial
    L = 20                 # Longueur du domaine
    N = 100                # Nombre de points
    dx = L / (N - 1)       # Pas spatial
    x = np.linspace(0, L, N)

    # 2. Paramètres temporels
    dt = 0.05
    T = 10
    nb_pas = int(T / dt)

    # 3. Condition initiale : Dirac en x = L/2, telle que somme(P0[i] Δx) = 1
    P0 = np.zeros(N)
    P0[N // 2] = 1 / dx
    P_euler = euler_diffusion(P0, D, dt, dx, nb_pas)
    P_analytique = solution_analytique_diffusion(x, T, D, dx, L / 2)

    plt.figure(figsize=(8, 5))
    plt.plot(x, P_euler[-1], "g", label="Euler implicite (numérique)")
    plt.plot(x, P_analytique, "r--", label="Solution analytique")
    plt.xlabel("x")
    plt.ylabel("P(x, T)")
    plt.title("Diffusion : Euler implicite vs solution analytique")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("diffusion_euler_vs_analytique.png", dpi=150)
    plt.show()


def calcul_ecart_type_erreur(D):
    """Quantifie la précision du schéma : écarts-types et erreurs entre solutions
    numérique et analytique, pour le D mesuré par les marches aléatoires."""
    # Domaine spatial
    L = 20
    N = 200
    dx = L / (N - 1)
    # Domaine temporel
    dt = 0.1
    T = 5.0
    nb_pas = int(T / dt)
    x = np.linspace(0, L, N)

    # Condition initiale : gaussienne très étroite (largeur 2Δx) pour approcher un Dirac
    epsilon = dx * 2
    P0 = np.exp(-(x - L / 2)**2 / (2 * epsilon**2))
    P0 = P0 / (np.sum(P0) * dx)  # Normalisation : somme(P0[i] Δx) = 1

    # Solution numérique au temps final T
    P_num = euler_diffusion(P0, D, dt, dx, nb_pas)[-1]
    # Solution analytique (reste gaussienne, d'écart-type √(2Dt))
    P_ana = solution_analytique_diffusion(x, T, D, dx, L / 2)

    # Écarts-types : mesure de l'étalement de chaque distribution
    moyenne_num = np.sum(x * P_num) / np.sum(P_num)
    ecart_type_num = np.sqrt(np.sum((x - moyenne_num)**2 * P_num) / np.sum(P_num))
    moyenne_ana = np.sum(x * P_ana) / np.sum(P_ana)
    ecart_type_ana = np.sqrt(np.sum((x - moyenne_ana)**2 * P_ana) / np.sum(P_ana))
    # Écart-type théorique pour la diffusion pure : σ = √(2Dt)
    ecart_type_theorique = np.sqrt(2 * D * T)

    # Erreur quadratique moyenne et erreur relative globale (norme L1).
    # Une erreur relative point par point exploserait dans les queues, où P_ana est quasi nulle.
    erreur_rms = np.sqrt(np.mean((P_num - P_ana)**2))
    erreur_relative = np.sum(np.abs(P_num - P_ana)) / np.sum(np.abs(P_ana))

    print("_" * 50)
    print("ANALYSE STATISTIQUE – DIFFUSION")
    print("_" * 50)
    print(f"Paramètre : D = {D:.6f}")
    print(f"Écart-type numérique           : {ecart_type_num:.6f}")
    print(f"Écart-type analytique          : {ecart_type_ana:.6f}")
    print(f"Écart-type théorique √(2DT)    : {ecart_type_theorique:.6f}")
    print(f"Erreur relative sur l'écart-type : "
          f"{abs(ecart_type_num - ecart_type_theorique) / ecart_type_theorique * 100:.4f} %")
    print(f"Erreur quadratique moyenne     : {erreur_rms:.6f}")
    print(f"Erreur relative globale (L1)   : {erreur_relative * 100:.4f} %")
    return ecart_type_num, ecart_type_theorique, erreur_rms


def calcul_coefficient_diffusion():
    """Calcule le coefficient de diffusion à partir de marches aléatoires sur réseau carré
    (référence : cours d'A. Torcini et A. Honecker)."""
    print("=" * 50)
    print("CALCUL DU COEFFICIENT DE DIFFUSION")
    print("-" * 50)

    pas = 100   # Chaque marcheur fait 100 pas
    N = 1000    # Nombre de marcheurs simulés

    # Distance quadratique moyenne à chaque pas
    d2moy = [0] * (pas + 1)
    for _ in range(N):
        x, y = 0, 0  # Départ au centre (0,0)
        for j in range(pas + 1):
            d2moy[j] += x * x + y * y
            if j < pas:
                # Déplacement aléatoire dans les 4 directions équiprobables
                direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                x += direction[0]
                y += direction[1]

    # Moyenne sur tous les marcheurs
    d2moy = np.array(d2moy) / N

    # Régression linéaire <r²> = 4 D n (marche 2D) : la pente vaut 4D
    nv = np.arange(pas + 1)
    pente, ordonnee = np.polyfit(nv, d2moy, 1)
    D_exp = pente / 4
    D_theorique = 0.25  # Réseau carré, pas unité, durée unité

    print("RÉSULTATS :")
    print(f"Distance quadratique moyenne finale : <r²> = {d2moy[-1]:.2f}")
    print(f"Coefficient de diffusion expérimental (régression) : D = {D_exp:.6f}")
    print(f"Coefficient de diffusion théorique : D = {D_theorique:.6f}")
    print(f"Erreur relative : {abs(D_exp - D_theorique) / D_theorique * 100:.2f} %")
    print("=" * 50)

    plt.figure(figsize=(10, 6))
    plt.plot(nv, d2moy, "b-", label="<r²> simulé (1 000 marcheurs)")
    plt.plot(nv, pente * nv + ordonnee, "k:", label=f"Régression linéaire (D = {D_exp:.4f})")
    plt.plot(nv, 4 * D_theorique * nv, "r--", label="<r²> théorique = 4Dn")
    plt.xlabel("Nombre de pas (n)")
    plt.ylabel("Distance quadratique moyenne <r²>")
    plt.title("Coefficient de diffusion – marche aléatoire 2D sur réseau carré")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("coefficient_diffusion.png", dpi=150)
    plt.show()

    return D_exp, D_theorique


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)  # Résultats reproductibles
    D_exp, D_theo = calcul_coefficient_diffusion()
    print("Calculs de diffusion en cours...")
    graphique_euler(D_exp)  # On utilise le D mesuré par simulation
    calcul_ecart_type_erreur(D_exp)
    print("\nFIN DU PROGRAMME")
