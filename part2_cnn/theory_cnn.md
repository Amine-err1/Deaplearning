# Partie II : Théorie sur les CNN et Vision par Ordinateur

## 1. Pourquoi MLP est mal adapté aux images

### 1.1 Problème de dimensionnalité
Une image de taille 28×28×3 (Fashion-MNIST est 28×28×1) contient 784 pixels. Si on l'aplatit (flatten) pour un MLP :
- Entrée : 784 dimensions
- Première couche cachée avec 1000 neurones : 784 × 1000 = 784,000 paramètres
- Cela crée un nombre massif de paramètres, favorisant l'overfitting

### 1.2 Perte de structure spatiale
Le MLP traite chaque pixel indépendamment, ignorant :
- La localité (pixels voisins sont corrélés)
- Les relations spatiales (haut/bas, gauche/droite)
- Les motifs géométriques (lignes, coins, textures)

### 1.3 Invariance par translation
Un MLP ne reconnaît pas qu'un motif déplacé dans l'image est le même motif. Chaque position nécessite un apprentissage séparé.

## 2. Localité

### 2.1 Principe
Les pixels proches spatialement sont fortement corrélés. Dans les images naturelles :
- Un pixel est fortement corrélé avec ses voisins immédiats
- La corrélation diminue avec la distance
- Les motifs locaux (bords, coins) sont répétés dans l'image

### 2.2 Implication pour les CNN
Les CNN exploitent cette localité en utilisant :
- **Filtres de petite taille** (ex: 3×3, 5×5)
- **Connexions locales** : chaque neurone ne voit qu'une petite région
- **Partage des poids** : le même filtre est appliqué partout

## 3. Partage des poids (Weight Sharing)

### 3.1 Définition
Le même ensemble de poids (filtre) est appliqué à toutes les positions de l'image.

### 3.2 Avantages
- **Réduction drastique des paramètres** : un filtre 3×3 = 9 paramètres au lieu de 784×1000
- **Invariance par translation** : le même détecteur fonctionne partout
- **Apprentissage de features universelles** : bords, textures, motifs

### 3.3 Exemple
Pour une image 28×28 avec un filtre 3×3 :
- Sans partage : 28×28×9 = 7,056 paramètres par position
- Avec partage : 9 paramètres seulement

## 4. Réceptive Field (Champ Récepteur)

### 4.1 Définition
La région de l'image d'entrée qui influence un neurone donné dans une couche profonde.

### 4.2 Calcul
Pour un CNN avec :
- Filtre de taille k
- Stride s
- Padding p

Le réceptive field augmente avec la profondeur :
- Couche 1 : RF = k
- Couche 2 : RF = k + (k-1) × s
- Couche n : RF = k + (k-1) × s × (n-1)

### 4.3 Importance
Un réceptive field plus grand permet de capturer :
- Des motifs plus complexes
- Des relations à plus longue distance
- Une hiérarchie de features (pixels → bords → motifs → objets)

## 5. Padding

### 5.1 Définition
Ajout de pixels autour de l'image pour contrôler la taille de sortie.

### 5.2 Types de padding
- **Valid** : pas de padding, sortie plus petite
- **Same** : padding pour garder la même taille
- **Full** : padding maximal, sortie plus grande

### 5.3 Calcul de la taille de sortie
Pour une entrée de taille H×W, filtre k×k, stride s, padding p :

```
H_out = floor((H - k + 2p) / s) + 1
W_out = floor((W - k + 2p) / s) + 1
```

### 5.4 Pourquoi utiliser padding ?
- **Garder la dimension** : facilite l'architecture
- **Préserver les bords** : les pixels aux bords sont moins traités sans padding
- **Empêcher la réduction trop rapide** : permet plus de couches

## 6. Stride

### 6.1 Définition
Le pas avec lequel le filtre se déplace sur l'image.

### 6.2 Effet sur la taille
- **Stride = 1** : sortie presque même taille
- **Stride = 2** : sortie divisée par 2 (downsampling)
- **Stride > 2** : réduction plus agressive

### 6.3 Compromis
- **Stride élevé** : moins de calculs, mais perte d'information
- **Stride faible** : plus de calculs, mais plus d'information préservée

## 7. Pooling

### 7.1 Objectif
Réduire la dimension spatiale tout en préservant l'information importante.

### 7.2 Types de pooling

**Max Pooling :**
- Prend le maximum dans chaque région
- Préserve les features les plus forts
- Insensible aux petites translations

**Average Pooling :**
- Prend la moyenne dans chaque région
- Lisse les features
- Moins sensible aux valeurs extrêmes

### 7.3 Avantages du pooling
- **Réduction de dimension** : moins de paramètres, moins de calcul
- **Invariance par translation** : petite translation ne change pas la sortie
- **Hiérarchie** : permet d'abstraire l'information

### 7.4 Calcul de la taille
Pour un pooling k×k avec stride s :

```
H_out = floor((H - k) / s) + 1
W_out = floor((W - k) / s) + 1
```

## 8. Convolution 1×1

### 8.1 Définition
Une convolution avec un filtre de taille 1×1.

### 8.2 Utilisation
- **Changement de canaux** : réduit ou augmente le nombre de canaux
- **Non-linéarité** : ajoute une transformation non-linéaire
- **Bottleneck** : réduit la dimensionnalité avant une opération coûteuse

### 8.3 Exemple
Entrée : H×W×C_in
Filtre 1×1 : C_out filtres
Sortie : H×W×C_out

### 8.4 Avantages
- Très peu de paramètres (C_in × C_out)
- Permet de contrôler la profondeur indépendamment de la taille spatiale
- Utilisé dans ResNet, Inception, MobileNet

## 9. LeNet

### 9.1 Architecture historique
Introduit par Yann LeCun en 1998 pour la reconnaissance de chiffres (MNIST).

### 9.2 Architecture originale
```
Input (32×32) → Conv1 (5×5, 6) → Pool (2×2) → 
Conv2 (5×5, 16) → Pool (2×2) → 
FC (120) → FC (84) → Output (10)
```

### 9.3 Caractéristiques
- **Convolutions** : extraction de features locaux
- **Pooling** : réduction de dimension et invariance
- **Fully Connected** : classification finale
- **Activation tanh** : fonction d'activation de l'époque

### 9.4 Influence
LeNet a établi les principes fondamentaux des CNN modernes :
- Alternance convolution/pooling
- Features hiérarchiques
- Partage des poids
- Réceptive field croissant

## 10. Calculs de Convolution

### 10.1 Corrélation croisée (Cross-Correlation)
En deep learning, on utilise la corrélation croisée (pas la vraie convolution mathématique) :

```
Output(i,j) = Σ Σ Input(i+m, j+n) × Kernel(m,n)
```

Où le kernel n'est pas retourné (contrairement à la convolution mathématique).

### 10.2 Exemple manuel
Input 3×3 :
```
1 2 3
4 5 6
7 8 9
```

Kernel 2×2 :
```
1 0
0 1
```

Convolution sans padding, stride=1 :
```
Position (0,0): 1×1 + 2×0 + 4×0 + 5×1 = 6
Position (0,1): 2×1 + 3×0 + 5×0 + 6×1 = 8
Position (1,0): 4×1 + 5×0 + 7×0 + 8×1 = 12
Position (1,1): 5×1 + 6×0 + 8×0 + 9×1 = 14
```

Sortie 2×2 :
```
6  8
12 14
```

### 10.3 Calcul des dimensions
Pour une entrée H×W×C_in avec K filtres de taille k×k, stride s, padding p :

**Sortie spatiale :**
```
H_out = floor((H - k + 2p) / s) + 1
W_out = floor((W - k + 2p) / s) + 1
```

**Sortie canaux :**
```
C_out = K
```

**Nombre de paramètres :**
```
Params = k × k × C_in × K + K (biais)
```

### 10.4 Exemple avec canaux multiples
Input : 28×28×3 (image RGB)
Filtre : 5×5×3, 16 filtres
Stride : 1, Padding : 0

Sortie : 24×24×16
Paramètres : 5×5×3×16 + 16 = 1216

## 11. Calculs de Pooling

### 11.1 Max Pooling
Pour une région k×k :
```
Output(i,j) = max{Input(i+m, j+n) pour m,n dans [0,k-1]}
```

### 11.2 Average Pooling
Pour une région k×k :
```
Output(i,j) = (1/k²) × Σ Σ Input(i+m, j+n)
```

### 11.3 Exemple
Input 4×4 :
```
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16
```

Max Pooling 2×2, stride=2 :
```
max(1,2,5,6) = 6     max(3,4,7,8) = 8
max(9,10,13,14) = 14 max(11,12,15,16) = 16
```

Sortie 2×2 :
```
6  8
14 16
```

### 11.4 Calcul des dimensions
Pour une entrée H×W avec pooling k×k, stride s :

```
H_out = floor((H - k) / s) + 1
W_out = floor((W - k) / s) + 1
```

## 12. Hiérarchie des Features

### 12.1 Organisation des features dans un CNN
- **Couches basses** : détectent des motifs simples (bords, coins, couleurs)
- **Couches moyennes** : combinent les motifs simples (textures, formes)
- **Couches hautes** : combinent les formes complexes (objets, parties)

### 12.2 Pourquoi ça marche
- **Compositionnalité** : les objets sont composés de motifs simples
- **Réutilisation** : les mêmes motifs simples sont réutilisés partout
- **Abstraction progressive** : on passe du détail au concept

### 12.3 Visualisation
Les feature maps des couches basses ressemblent à des détecteurs de bords, tandis que celles des couches hautes ressemblent à des parties d'objets.
