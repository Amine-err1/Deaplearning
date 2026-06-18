# Partie III : Théorie sur les RNN, LSTM, GRU et Seq2Seq

## 1. Modèle de Langage

### 1.1 Définition
Un modèle de langage est un modèle probabiliste qui assigne une probabilité à une séquence de mots.

### 1.2 Objectif
Apprendre P(w₁, w₂, ..., wₙ), la probabilité jointe d'une séquence de mots.

### 1.3 Applications
- Prédiction de texte
- Traduction automatique
- Reconnaissance vocale
- Analyse de sentiments
- Génération de texte

## 2. Règle de Chaîne (Chain Rule)

### 2.1 Définition
La règle de chaîne permet de décomposer la probabilité jointe en produit de probabilités conditionnelles :

**P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)**

### 2.2 Interprétation
La probabilité d'une séquence est le produit des probabilités de chaque mot sachant tous les mots précédents.

### 2.3 Approximation
En pratique, on ne peut pas conditionner sur tous les mots précédents (trop de contexte). On utilise une fenêtre de taille fixe ou un modèle récurrent.

## 3. Probabilités Conditionnelles

### 3.1 Définition
P(wₙ|w₁, ..., wₙ₋₁) est la probabilité du mot wₙ sachant le contexte w₁, ..., wₙ₋₁.

### 3.2 Estimation
Dans un modèle de langage, cette probabilité est estimée par :
- **n-grammes** : P(wₙ|wₙ₋ₙ₊₁, ..., wₙ₋₁)
- **RNN** : P(wₙ|hₙ₋₁) où hₙ₋₁ est l'état caché résumant l'historique

### 3.3 Log-probabilités
En pratique, on travaille avec les log-probabilités pour éviter la sous-overflow :

**log P(w₁, ..., wₙ) = Σ log P(wᵢ|contexte)**

## 4. Perplexité

### 4.1 Définition
La perplexité mesure la "surprise" d'un modèle de langage. Plus elle est basse, meilleur est le modèle.

**Perplexité = 2^(-1/n × Σ log₂ P(wᵢ|contexte))**

### 4.2 Interprétation
- Perplexité = 10 : Le modèle est aussi "surpris" que s'il devait choisir entre 10 mots équiprobables
- Perplexité = 1 : Le modèle prédit parfaitement la séquence
- Perplexité élevée : Le modèle est souvent surpris (mauvais)

### 4.3 Relation avec l'entropie
La perplexité est l'exponentielle de l'entropie croisée :

**Perplexité = exp(CrossEntropy)**

## 5. RNN (Recurrent Neural Network)

### 5.1 Architecture
Un RNN traite les séquences pas à pas en maintenant un état caché :

**hₜ = f(Wₕₕhₜ₋₁ + Wₓₕxₜ + bₕ)**
**yₜ = g(Wₕᵧhₜ + bᵧ)**

Où :
- hₜ : état caché au temps t
- xₜ : entrée au temps t
- yₜ : sortie au temps t
- Wₕₕ, Wₓₕ, Wₕᵧ : matrices de poids
- f, g : fonctions d'activation (tanh, softmax)

### 5.2 Avantages
- Traite des séquences de longueur variable
- Partage des paramètres à travers le temps
- Mémoire de l'historique via l'état caché

### 5.3 Limitations
- **Vanishing gradients** : Les gradients deviennent trop petits pour les longues séquences
- **Exploding gradients** : Les gradients deviennent trop grands, causant l'instabilité
- **Mémoire limitée** : L'état caché ne peut stocker que peu d'information à long terme

## 6. LSTM (Long Short-Term Memory)

### 6.1 Motivation
Le LSTM a été introduit pour résoudre le problème du vanishing gradient dans les RNN simples.

### 6.2 Architecture
Le LSTM introduit une cellule de mémoire (c) et trois portes :

**Porte d'oubli (Forget Gate) :**
fₜ = σ(Wₓf xₜ + Wₕf hₜ₋₁ + b_f)

**Porte d'entrée (Input Gate) :**
iₜ = σ(Wₓᵢ xₜ + Wₕᵢ hₜ₋₁ + bᵢ)
gₜ = tanh(Wₓg xₜ + Wₕg hₜ₋₁ + b_g)

**Mise à jour de la cellule :**
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ

**Porte de sortie (Output Gate) :**
oₜ = σ(Wₓo xₜ + Wₕo hₜ₋₁ + b_o)
hₜ = oₜ ⊙ tanh(cₜ)

Où :
- σ : sigmoïde
- ⊙ : multiplication élément par élément
- cₜ : cellule de mémoire
- hₜ : état caché

### 6.3 Intuition
- **Forget gate** : Décide quoi oublier de la mémoire
- **Input gate** : Décide quoi ajouter à la mémoire
- **Output gate** : Décide quoi lire de la mémoire

### 6.4 Avantages sur RNN simple
- Gradients qui peuvent traverser de longues séquences sans s'évanouir
- Capacité à apprendre des dépendances à long terme
- Contrôle explicite du flux d'information

## 7. GRU (Gated Recurrent Unit)

### 7.1 Motivation
Le GRU est une simplification du LSTM avec moins de paramètres mais des performances comparables.

### 7.2 Architecture
Le GRU a deux portes au lieu de trois :

**Porte de mise à jour (Update Gate) :**
zₜ = σ(Wₓz xₜ + Wₕz hₜ₋₁ + b_z)

**Porte de réinitialisation (Reset Gate) :**
rₜ = σ(Wₓr xₜ + Wₕr hₜ₋₁ + b_r)

**État candidat :**
h̃ₜ = tanh(Wₓh xₜ + Wₕh (rₜ ⊙ hₜ₋₁) + b_h)

**Mise à jour de l'état caché :**
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ

### 7.3 Intuition
- **Update gate** : Contrôle combien de l'état précédent garder
- **Reset gate** : Contrôle combien de l'état précédent oublier

### 7.4 Comparaison LSTM vs GRU
- **GRU** : Moins de paramètres, plus rapide, souvent comparable en performance
- **LSTM** : Plus expressif, meilleur pour des tâches complexes nécessitant plus de mémoire

## 8. Seq2Seq (Sequence-to-Sequence)

### 8.1 Architecture
Un modèle Seq2Seq transforme une séquence en une autre séquence :

**Encodeur :** Traite la séquence d'entrée et produit un contexte
**Décodeur :** Génère la séquence de sortie à partir du contexte

### 8.2 Architecture de base
```
Encodeur (RNN/LSTM/GRU) → Vecteur contexte → Décodeur (RNN/LSTM/GRU)
```

### 8.3 Applications
- Traduction automatique
- Résumé de texte
- Dialogue (chatbot)
- Génération de code

## 9. Teacher Forcing

### 9.1 Définition
Teacher forcing est une technique d'entraînement où le décodeur reçoit le vrai token précédent au lieu de sa propre prédiction.

### 9.2 Pendant l'entraînement
- **Avec teacher forcing** : yₜ₊₁ = vrai token au temps t+1
- **Sans teacher forcing** : yₜ₊₁ = prédiction du modèle au temps t+1

### 9.3 Avantages
- Accélère la convergence
- Stabilise l'entraînement
- Évite l'accumulation d'erreurs

### 9.4 Inconvénients
- Exposition bias : Le modèle n'apprend pas à corriger ses propres erreurs
- Discrepancy entre entraînement et inférence

### 9.5 Scheduled Sampling
Compromis : utiliser teacher forcing avec une probabilité qui décroît au cours de l'entraînement.

## 10. Beam Search

### 10.1 Définition
Beam search est une technique de décodage qui explore plusieurs hypothèses en parallèle.

### 10.2 Algorithme
1. Initialiser avec le token de début
2. À chaque étape, garder les k meilleures séquences partielles (beam width = k)
3. Étendre chaque séquence avec tous les tokens possibles
4. Garder les k meilleures nouvelles séquences
5. Répéter jusqu'à la fin de séquence

### 10.3 Avantages sur Greedy Decoding
- Greedy : choisit toujours le meilleur token à chaque étape (myope)
- Beam search : considère plusieurs chemins, trouve souvent une meilleure séquence globale

### 10.4 Score normalisé
Pour éviter de favoriser les séquences courtes, on normalise par la longueur :

**Score = (Σ log P(wᵢ)) / (longueur)^α**

Où α est un facteur de longueur (ex: 0.6).

## 11. BLEU Score

### 11.1 Définition
BLEU (Bilingual Evaluation Understudy) mesure la qualité d'une traduction en comparant avec des références.

### 11.2 Calcul
**BLEU = BP × exp(Σ wₙ log pₙ)**

Où :
- BP : Penalty de brièveté (pour pénaliser les traductions trop courtes)
- pₙ : précision des n-grammes (unigrammes, bigrammes, trigrammes, 4-grammes)
- wₙ : poids (généralement 0.25 pour chaque n)

### 11.3 Précision des n-grammes
pₙ = (nombre de n-grammes corrects) / (nombre total de n-grammes dans la prédiction)

### 11.4 Interprétation
- BLEU = 0 : Aucun n-gramme correct
- BLEU = 1 : Traduction parfaite
- BLEU > 0.3 : Traduction acceptable
- BLEU > 0.5 : Bonne traduction

## 12. BPTT (Backpropagation Through Time)

### 12.1 Définition
BPTT est l'algorithme de rétropropagation adapté aux RNN. Il "déplie" le RNN dans le temps et applique la rétropropagation standard.

### 12.2 Processus
1. Déplier le RNN sur T étapes de temps
2. Calculer la forward pass sur toute la séquence
3. Calculer la loss à chaque étape
4. Rétropropager les gradients à travers le temps
5. Mettre à jour les paramètres partagés

### 12.3 Complexité
- Temps : O(T × n_paramètres)
- Mémoire : O(T) pour stocker les états intermédiaires

### 12.4 Truncated BPTT
Pour les très longues séquences, on tronque la BPTT à une longueur fixe pour économiser la mémoire.

## 13. Gradient Clipping

### 13.1 Problème
Dans les RNN, les gradients peuvent exploser (devenir très grands), causant :
- Instabilité numérique
- Paramètres qui divergent
- NaN dans les calculs

### 13.2 Solution
Le gradient clipping limite la norme des gradients :

**Si ||g|| > threshold : g = g × (threshold / ||g||)**

### 13.3 Types de clipping
- **Norm clipping** : Limite la norme du gradient (le plus courant)
- **Value clipping** : Limite chaque composante individuellement

### 13.4 Seuil typique
Valeurs courantes : 0.5, 1.0, 5.0

## 14. Bidirectional RNN

### 14.1 Définition
Un RNN bidirectionnel traite la séquence dans les deux directions (avant et arrière).

### 14.2 Architecture
```
RNN avant : hₜ^→ = f(xₜ, hₜ₋₁^→)
RNN arrière : hₜ^← = f(xₜ, hₜ₊₁^←)
État combiné : hₜ = [hₜ^→, hₜ^←]
```

### 14.3 Avantages
- Accès au contexte futur
- Meilleur pour les tâches où le contexte complet est disponible (ex: classification)

### 14.4 Limitations
- Ne peut pas être utilisé pour la génération (pas de contexte futur pendant l'inférence)
- Double les paramètres

## 15. Attention Mechanism

### 15.1 Motivation
Dans Seq2Seq classique, toute la séquence d'entrée est compressée dans un vecteur contexte fixe, ce qui est limitant pour les longues séquences.

### 15.2 Principe
L'attention permet au décodeur de "regarder" différentes parties de la séquence d'entrée à chaque étape de décodage.

### 15.3 Calcul
**Score d'attention :**
eₜᵢ = score(hₜ₋₁, hᵢ)

**Poids d'attention :**
αₜᵢ = softmax(eₜᵢ)

**Contexte :**
cₜ = Σ αₜᵢ hᵢ

**État du décodeur :**
hₜ = f(cₜ, hₜ₋₁, yₜ₋₁)

### 15.4 Types d'attention
- **Additive attention** (Bahdanau) : score = vᵀ tanh(W₁hₜ₋₁ + W₂hᵢ)
- **Multiplicative attention** (Luong) : score = hₜ₋₁ᵀ W hᵢ
- **Self-attention** (Transformer) : attention entre tokens de la même séquence

## 16. Embeddings

### 16.1 Définition
Les embeddings sont des représentations vectorielles denses de mots (ou tokens).

### 16.2 Avantages sur One-Hot
- **Dimensionnalité réduite** : 300 vs 50,000 (taille du vocabulaire)
- **Capture la sémantique** : Mots similaires ont des embeddings similaires
- **Apprenables** : Les embeddings sont optimisés pendant l'entraînement

### 16.3 Types
- **Random** : Initialisés aléatoirement et appris
- **Pre-trained** : Word2Vec, GloVe, FastText (transfert learning)
- **Contextual** : BERT, GPT (embeddings dépendent du contexte)

## 17. Padding et Masquage

### 17.1 Padding
Les séquences dans un batch doivent avoir la même longueur. On ajoute des tokens de padding (<PAD>) aux séquences courtes.

### 17.2 Masquage
Le masque indique quels tokens sont réels et quels sont du padding pour :
- Ignorer le padding dans le calcul de la loss
- Empêcher le padding d'influencer l'attention

### 17.3 Masque d'attention
Dans les modèles avec attention, le masque empêche d'atténuer sur les tokens de padding.

## 18. Mini-batch pour Séquences

### 18.1 Défis
- Les séquences ont des longueurs variables
- Le padding gaspille de la mémoire
- L'ordre des séquences peut influencer l'entraînement

### 18.2 Solutions
- **Batching par longueur** : Grouper les séquences de longueurs similaires
- **Dynamic batching** : Adapter la taille du batch selon la longueur
- **Bucketing** : Diviser les séquences en buckets de longueurs

### 18.3 Packing
PyTorch offre `pack_padded_sequence` pour ignorer efficacement le padding dans les RNN.
