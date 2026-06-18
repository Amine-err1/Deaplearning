# Partie I : Théorie sur les Perceptrons et MLP

## 1. Le Perceptron

### 1.1 Définition
Le perceptron est le neurone artificiel le plus simple, introduit par Frank Rosenblatt en 1957. Il s'agit d'un classificateur binaire linéaire qui apprend par ajustement des poids.

### 1.2 Mathématiques du Perceptron

Pour un vecteur d'entrée **x** ∈ ℝⁿ, le perceptron calcule :

**z = w·x + b = Σᵢ wᵢxᵢ + b**

Où :
- **w** = (w₁, w₂, ..., wₙ) est le vecteur de poids
- b est le biais
- **x** = (x₁, x₂, ..., xₙ) est le vecteur d'entrée

La sortie est donnée par la fonction d'activation :

**ŷ = σ(z)**

Où σ est une fonction d'activation (sigmoïde, ReLU, etc.)

### 1.3 Limitations du Perceptron
Le perceptron simple ne peut résoudre que des problèmes linéairement séparables (ex: XOR ne peut pas être résolu).

## 2. Multi-Layer Perceptron (MLP)

### 2.1 Architecture
Un MLP est un réseau de neurones à couches entièrement connectées (fully-connected) :

```
Entrée → Couche cachée 1 → Couche cachée 2 → ... → Sortie
```

Chaque couche est composée de plusieurs neurones. Les connexions sont totales entre les couches adjacentes.

### 2.2 Propagation avant (Forward Propagation)

Pour un MLP avec L couches cachées :

**h⁰ = x** (entrée)

**hˡ = σˡ(Wˡhˡ⁻¹ + bˡ)** pour l = 1, 2, ..., L

**ŷ = σᵏ⁺¹(Wᵏ⁺¹hᴸ + bᵏ⁺¹)** (sortie)

Où :
- hˡ est la sortie de la couche l
- Wˡ est la matrice de poids de la couche l
- bˡ est le vecteur de biais de la couche l
- σˡ est la fonction d'activation de la couche l

### 2.3 Fonctions d'activation courantes

**Sigmoïde :**
σ(z) = 1 / (1 + e⁻ˣ)
σ'(z) = σ(z)(1 - σ(z))

**Tanh :**
tanh(z) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
tanh'(z) = 1 - tanh²(z)

**ReLU (Rectified Linear Unit) :**
ReLU(z) = max(0, z)
ReLU'(z) = 1 si z > 0, 0 sinon

**Leaky ReLU :**
LeakyReLU(z) = max(αz, z) avec α petit (ex: 0.01)

## 3. nn.Module dans PyTorch

### 3.1 Concept
`nn.Module` est la classe de base pour tous les réseaux de neurones dans PyTorch. Elle fournit :
- Gestion automatique des paramètres
- Méthodes pour déplacer le modèle sur CPU/GPU
- Méthodes pour sauvegarder/charger le modèle

### 3.2 Structure d'un module personnalisé

```python
import torch.nn as nn

class MonMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MonMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

### 3.3 Avantages
- Flexibilité totale dans l'architecture
- Possibilité d'ajouter des logiques personnalisées
- Héritage et composition possibles

## 4. nn.Sequential dans PyTorch

### 4.1 Concept
`nn.Sequential` est un conteneur qui exécute les modules dans l'ordre séquentiel.

### 4.2 Utilisation

```python
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
```

### 4.3 Avantages
- Syntaxe concise
- Idéal pour des architectures linéaires simples
- Moins de code boilerplate

### 4.4 Limitations
- Pas de logique conditionnelle
- Pas de branches multiples
- Difficile pour des architectures complexes

## 5. Paramètres du réseau

### 5.1 Définition
Les paramètres d'un réseau sont les poids (W) et les biais (b) qui sont appris pendant l'entraînement.

### 5.2 Accès aux paramètres

```python
# Itérer sur tous les paramètres
for param in model.parameters():
    print(param.shape)

# Accéder aux paramètres nommés
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 5.3 Nombre de paramètres
Le nombre total de paramètres est la somme de tous les éléments dans toutes les matrices de poids et vecteurs de biais.

Pour une couche linéaire de dimension n_in → n_out :
- Poids : n_in × n_out
- Biais : n_out
- Total : n_in × n_out + n_out

## 6. Gradients

### 6.1 Définition
Le gradient d'une fonction f(x) est le vecteur des dérivées partielles :

∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

### 6.2 Rôle dans l'entraînement
Les gradients indiquent la direction dans laquelle modifier les paramètres pour minimiser la fonction de perte.

### 6.3 Calcul automatique (Autograd)
PyTorch calcule automatiquement les gradients par rétropropagation :

```python
loss = criterion(output, target)
loss.backward()  # Calcule les gradients
optimizer.step()  # Met à jour les paramètres
```

## 7. Rétropropagation (Backpropagation)

### 7.1 Principe
La rétropropagation est l'algorithme qui calcule efficacement les gradients de la fonction de perte par rapport à tous les paramètres du réseau en utilisant la règle de chaîne.

### 7.2 Règle de chaîne
Pour une fonction composée f(g(x)) :

∂f/∂x = (∂f/∂g) × (∂g/∂x)

### 7.3 Algorithme
1. Propagation avant : calculer la sortie
2. Calculer la perte
3. Propagation arrière : calculer les gradients couche par couche
4. Mise à jour des paramètres

### 7.4 Pourquoi c'est efficace
Au lieu de recalculer chaque gradient indépendamment, la rétropropagation réutilise les calculs intermédiaires, réduisant la complexité de O(n²) à O(n).

## 8. state_dict

### 8.1 Définition
`state_dict` est un dictionnaire Python qui mappe chaque couche à son tenseur de paramètres.

### 8.2 Structure

```python
state_dict = {
    'layer1.weight': tensor(...),
    'layer1.bias': tensor(...),
    'layer2.weight': tensor(...),
    'layer2.bias': tensor(...)
}
```

### 8.3 Sauvegarde et chargement

```python
# Sauvegarder
torch.save(model.state_dict(), 'model.pth')

# Charger
model = MonMLP(...)
model.load_state_dict(torch.load('model.pth'))
```

### 8.4 Avantages
- Sépare les paramètres de l'architecture
- Permet de sauvegarder uniquement les poids (plus léger)
- Flexibilité pour charger des poids dans des architectures différentes

## 9. CPU vs GPU

### 9.1 Pourquoi utiliser GPU ?
- **Parallélisme** : Les GPU ont des milliers de cœurs vs quelques dizaines pour CPU
- **Calcul matriciel** : Les opérations sur tenseurs sont massivement parallélisables
- **Vitesse** : 10-100x plus rapide pour l'entraînement de réseaux profonds

### 9.2 Transfert CPU ↔ GPU

```python
# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Déplacer le modèle
model = model.to(device)

# Déplacer les données
x = x.to(device)
y = y.to(device)
```

### 9.3 Considérations
- **Mémoire** : Les GPU ont moins de mémoire que les CPU
- **Transfert** : Le transfert CPU→GPU a un coût, minimiser les transferts
- **Batch size** : Adapter le batch size à la mémoire GPU

### 9.4 Mixed Precision
Utiliser des calculs en précision mixte (FP16 + FP32) pour :
- Réduire l'usage mémoire
- Accélérer les calculs
- Avec une perte négligeable de précision

## 10. Initialisation des poids

### 10.1 Importance
Une mauvaise initialisation peut causer :
- **Vanishing gradients** : gradients trop petits, apprentissage bloqué
- **Exploding gradients** : gradients trop grands, instabilité numérique

### 10.2 Méthodes d'initialisation

**Zéro/Constante :**
```python
nn.init.constant_(layer.weight, 0)
```
- ❌ **Problème** : Tous les neurones apprennent la même chose

**Gaussienne (Normal) :**
```python
nn.init.normal_(layer.weight, mean=0, std=0.01)
```
- Mieux que zéro mais peut encore causer des problèmes

**Xavier/Glorot :**
```python
nn.init.xavier_uniform_(layer.weight)
# ou
nn.init.xavier_normal_(layer.weight)
```
- Conçu pour les fonctions d'activation sigmoïde/tanh
- Var = 2 / (n_in + n_out)

**He/Kaiming :**
```python
nn.init.kaiming_uniform_(layer.weight, mode='fan_in')
# ou
nn.init.kaiming_normal_(layer.weight, mode='fan_in')
```
- Conçu pour ReLU et ses variantes
- Var = 2 / n_in

### 10.3 Pourquoi ces méthodes ?
L'objectif est de maintenir la variance des activations et des gradients constants à travers les couches, évitant ainsi les vanishing/exploding gradients.
