"""
Projet Numérique L3 - Dynamique Moléculaire
Semy El Karra
Description : Simulation de N particules interagissant via le potentiel de Lennard-Jones.
Intégration par l'algorithme de Verlet des vitesses.
Ensemble Microcanonique (N, V, E) avec Conditions aux Limites Périodiques.

1. Modules externes :
   - numpy (np) : Pour la manipulation des tableaux de données (arrays).
   - matplotlib.pyplot (plt) : Pour le tracé et la sauvegarde des figures.

2. Fonctions de la bibliothèque NumPy exploitées :
   - np.zeros() : Allocation mémoire pour les matrices (positions, vitesses, forces).
   - np.random.standard_normal() : Tirage stochastique pour les vitesses initiales (M-B).
   - np.mean() / np.sum() : Calculs statistiques (vitesse du centre de masse, énergie cinétique).
   - np.round() : Outil mathématique pour la convention de l'image minimum (CLP).
   - array.dot() : Produit scalaire pour le calcul rapide des distances au carré (r^2).
   - np.copy() : Copie profonde du tenseur des forces pour le pas d'intégration Verlet.
   - np.sqrt() / np.exp() : Opérations mathématiques basiques (modules, exponentielle).
   - np.linspace() : Génération du vecteur d'abscisses pour la courbe théorique.

3. Outils Python  :
   - Boucles itératives (for) et structures conditionnelles (if).
   - Opérateurs mathématiques : division entière (//) et modulo (%) pour le réseau et les CLP.
   - Listes dynamiques (.append(), .extend()) pour le stockage de l'historique thermodynamique.

4. Algo Cours:
   - forces_and_ep(posi, L) : Fonction maîtresse calculant le gradient du potentiel 
     de Lennard-Jones et l'énergie potentielle, optimisée via la 3ème loi de Newton.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # Résultats reproductibles

# --- Paramètres de la simulation ---
"""Pour éviter des forces de répulsion infinies au départ (si r -> 0), on place les 
atomes sur un réseau cristallin régulier.
La formule `int(Natom**(1/DIM) + 0.99)` calcule la racine carrée (en 2D) du nombre 
d'atomes et simule la fonction "plafond" (ceil). Cela garantit la création d'une 
grille juste assez grande (ex: 8x8) pour contenir toutes nos particules."""


DIM=2                             # Dimension de l'espace (2D)
Natom=64                          # Nombre total de particules
d0=1.2                            # Paramètre de maille (distance initiale entre atomes)
cotereseau=int(Natom**(1/DIM)+.99)# Nombre de particules par côté du réseau initial (ex: 8x8)
TailleBoite=cotereseau*d0       # Taille de la boîte de simulation (L)
Ratom=0.3                         # Rayon des particules (purement pour l'affichage)
Vi_Max=0.5                        # Facteur d'échelle pour les vitesses initiales
h=0.01                            # Pas de temps d'intégration (dt)
itmax=3000                        # Nombre total d'itérations de la boucle principale
t_thermalisation=1000             # Temps laissé au système pour atteindre l'équilibre thermodynamique




# --- Initialisation ---
posi= np.zeros((Natom, DIM))         # Positions
vitess= np.zeros((Natom, DIM))       # Vitesses
force= np.zeros((Natom, DIM))          # Forces

# Positions initiales sur un réseau régulier (pour éviter les chevauchements)
for i in range(Natom):
    for k in range(DIM):
        posi[i,k]=(i//cotereseau**k) % cotereseau
        posi[i,k]=posi[i,k] * d0

# --- Vitesses initiales et conservation de la quantité de mouvement ---
# Tirage selon une loi normale pour approximer Maxwell-Boltzmann à l'instant t=0
vitess = Vi_Max*np.random.standard_normal((Natom, DIM))

"""On centre les vitesses en soustrayant la moyenne globale.
Cela annule la vitesse du centre de masse (quantité de mouvement totale = 0).
Sans cela, la boîte de simulation entière dériverait, ce qui fausserait 
la mesure de la température (énergie cinétique interne)."""
vitess -= np.mean(vitess, axis=0)

# --- Fonction de calcul des forces et énergie potentielle ---
def forces_and_ep(posi,L):
    force = np.zeros((Natom,DIM))
    Ep=0.0
    # Rayon de coupure (rc) au carré. Pour r > 3*sigma, l'attraction de van der Waals 
    # devient négligeable. L'ignorer permet de gagner énormément de temps de calcul.
    rc_carre = (3.0)**2 
    
    #Complexité O(N^2) divisée par 2
    # La double boucle (j partant de i+1) exploite la 3ème loi de Newton (F_ij = -F_ji).
    # On itère uniquement sur les paires uniques pour ne calculer la force qu'une seule fois.
    for i in range(Natom-1):
        for j in range(i+1, Natom):
            dr= posi[i]-posi[j]
            
            #Convention de l'Image Minimum (CLP) 
            # Au lieu d'utiliser des conditions complexes (if dr > L/2...), cette ligne 
            # ramène instantanément la distance vectorielle 'dr' 
            # dans l'intervalle [-L/2, L/2]. Cela simule la périodicité de la boîte.
            dr=dr-L*np.round(dr/L)
            # Distance au carré (produit scalaire)
            r2=dr.dot(dr)
            
            if r2 < rc_carre and r2 > 0.1: 
                #Optimisation du calcul de Lennard-Jones ---
                # On évite délibérément de calculer r = sqrt(r2) car la racine carrée 
                # est une opération très coûteuse pour le processeur.
                # En unités réduites (sigma=1), le terme en (sigma/r)^6 s'écrit simplement (r2)^-3.
                rm6=r2**-3               
                # Force de Lennard-Jones
                fij=(48.0/r2*(rm6-0.5)*rm6)*dr 

                # Action-Réaction
                force[i]+=fij
                force[j]-=fij
                # Énergie potentielle de Lennard-Jones
                Ep+=4.0*(rm6**2-rm6)
                
    return force, Ep




# --- Boucle de Simulation (Verlet des vitesses) ---
time_steps = []
Ec_list, Ep_list, Etot_list = [], [], []
# Pour suivre 3 particules spécifiques
traj_p1_x, traj_p1_y = [], []
traj_p2_x, traj_p2_y = [], []
traj_p3_x, traj_p3_y = [], []

# Échantillon de vitesses pour l'étude approfondie (après thermalisation)
vitesses_echantillon=[]

print("Début de la simulation. Veuillez patienter un instant...")
# Premier calcul des forces
force, Ep = forces_and_ep(posi, TailleBoite)

for it in range(itmax):
    # ---  Mise à jour des positions ---
    # r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2 (avec m=1, donc a=force)
    posi=posi+vitess*h+0.5* force*h**2
    
    # Repliement périodique (Conditions aux Limites) ---
    # L'opérateur modulo (%) est d'une efficacité redoutable ici : si une particule 
    # sort par la droite (pos > L), elle réapparaît à gauche (pos % L). Idem pour 
    # les valeurs négatives. Cela maintient une densité constante sans murs physiques.
    posi=posi%TailleBoite
    
    # 2. Calcul des nouvelles forces avec les nouvelles positions
    force_new, Ep = forces_and_ep(posi, TailleBoite)
    
    #3 : Mise à jour des vitesses
    # v(t+dt) = v(t) + 0.5 * [a(t) + a(t+dt)] * dt
    # C'est ce qui rend l'algorithme symplectique. Utiliser la moyenne 
    # des forces entre l'instant t et t+dt garantit une excellente conservation 
    # de l'énergie mécanique sur le très long terme.
    vitess=vitess+0.5*(force + force_new)*h
    
    # Mise à jour des forces pour la prochaine itération
    force = np.copy(force_new)
    
    # --- Mesures Thermodynamiques ---
    # Énergie cinétique = 1/2 * m * v^2 (avec m=1)
    Ec = 0.5 * np.sum(vitess**2)
    Etot = Ec + Ep
    
    # Enregistrement à intervalles réguliers (décimation pour ne pas saturer la mémoire)
    if it % 10 == 0:
        time_steps.append(it*h)
        Ec_list.append(Ec)
        Ep_list.append(Ep)
        Etot_list.append(Etot)
        
        # Enregistrement des trajectoires (sans sauts liés aux frontières périodiques)
        # On enregistre les vraies positions pour le tracé
        traj_p1_x.append(posi[0, 0]); traj_p1_y.append(posi[0, 1])
        traj_p2_x.append(posi[1, 0]); traj_p2_y.append(posi[1, 1])
        traj_p3_x.append(posi[2, 0]); traj_p3_y.append(posi[2, 1])
        
    # Acquisition à l'équilibre Ergodicité ---
    # On ne collecte les vitesses pour l'histogramme de Maxwell-Boltzmann QU'APRÈS 
    # le temps de thermalisation (t_thermalisation). L'état initial transitoire 
    # (la "fonte" du cristal) ne doit pas polluer nos statistiques d'équilibre.
    if it >t_thermalisation and it % 5==0:
        v_magnitudes=np.sqrt(np.sum(vitess**2,axis=1))
        vitesses_echantillon.extend(v_magnitudes)

print("Simulation terminée. Génération des graphiques...")

# Calcul de la Température moyenne (En 2D: <Ec> = N * kb * T / 2 -> T = <Ec> / N, kb=1)
Ec_moyen = np.mean(Ec_list[int(t_thermalisation/10):])
T_moyen = Ec_moyen / Natom

# ==========================================
# 1. Graphique des Énergies (Validation)
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(time_steps, Ec_list, label='Énergie Cinétique $E_k$', color='r', alpha=0.8)
plt.plot(time_steps, Ep_list, label='Énergie Potentielle $E_p$', color='b', alpha=0.8)
plt.plot(time_steps, Etot_list, label='Énergie Totale $E_{tot}$', color='k', linewidth=2)
plt.xlabel("Temps (t)")
plt.ylabel("Énergie")
plt.title("Évolution des énergies (Ensemble NVE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("energies.png")
plt.show()

# ==========================================
# 2. Graphique des Trajectoires
# ==========================================
def couper_sauts(tx, ty, L):
    """Insère une coupure (NaN) quand la particule traverse un bord de la boîte,
    pour ne pas tracer de fausse ligne d'un côté à l'autre (conditions périodiques)."""
    tx, ty = np.array(tx, dtype=float), np.array(ty, dtype=float)
    saut = (np.abs(np.diff(tx)) > L / 2) | (np.abs(np.diff(ty)) > L / 2)
    idx = np.where(saut)[0] + 1
    return np.insert(tx, idx, np.nan), np.insert(ty, idx, np.nan)

plt.figure(figsize=(7, 7))
plt.plot(*couper_sauts(traj_p1_x, traj_p1_y, TailleBoite), label='Particule 1', lw=1.5)
plt.plot(*couper_sauts(traj_p2_x, traj_p2_y, TailleBoite), label='Particule 2', lw=1.5)
plt.plot(*couper_sauts(traj_p3_x, traj_p3_y, TailleBoite), label='Particule 3', lw=1.5)
# Affichage des autres particules à la dernière étape
plt.scatter(posi[3:, 0], posi[3:, 1], c='gray', s=20, alpha=0.5, label='Bain (t_final)')
plt.xlim(0, TailleBoite)
plt.ylim(0, TailleBoite)
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.title("Trajectoires de 3 particules dans la boîte (CLP)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("trajectoires.png")
plt.show()

# ==========================================
# 3. Étude : Distribution des Vitesses
# ==========================================
plt.figure(figsize=(8, 5))
# Histogramme des vitesses simulées (densité=True pour normaliser)
count, bins, ignored = plt.hist(vitesses_echantillon, bins=40, density=True, 
                                alpha=0.6, color='teal', label='Simulation (Histogramme)')

# Courbe théorique de Maxwell-Boltzmann en 2D
# f(v) = (m / kb*T) * v * exp(-m*v^2 / 2*kb*T) avec m=1, kb=1
v_theorie = np.linspace(0, max(vitesses_echantillon), 100)
f_v = (v_theorie / T_moyen) * np.exp(- (v_theorie**2) / (2 * T_moyen))

plt.plot(v_theorie, f_v, 'k--', linewidth=2, label=f'Théorie M-B (T={T_moyen:.2f})')
plt.xlabel('Module de la vitesse $v$')
plt.ylabel('Densité de probabilité $f(v)$')
plt.title('Distribution des vitesses à l\'équilibre (Maxwell-Boltzmann)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vitesses.png")
plt.show()