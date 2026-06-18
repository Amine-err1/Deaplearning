# Résumé du Projet Deep Learning

## Vue d'ensemble

Ce projet complet de Deep Learning a été réalisé dans le cadre d'un projet universitaire, couvrant trois familles d'architectures fondamentales : MLP, CNN et RNN/LSTM/GRU/Seq2Seq. Chaque partie inclut une étude théorique détaillée, une implémentation PyTorch complète, des expérimentations comparatives et une analyse critique.

## Structure du Projet

```
deeplearning/
├── part1_mlp/                    # Partie I : MLP pour Breast Cancer Wisconsin
│   ├── theory_mlp.md            # Théorie détaillée sur perceptrons et MLP
│   ├── mlp_breast_cancer.ipynb  # Notebook principal avec implémentation
│   └── figures/                 # Visualisations générées
├── part2_cnn/                    # Partie II : CNN pour Fashion-MNIST
│   ├── theory_cnn.md            # Théorie détaillée sur CNN et vision par ordinateur
│   ├── cnn_fashion_mnist.ipynb  # Notebook principal avec implémentation
│   └── figures/                 # Visualisations générées
├── part3_rnn/                    # Partie III : RNN/LSTM/GRU pour IMDb
│   ├── theory_rnn.md            # Théorie détaillée sur modèles de langage et RNN
│   ├── rnn_imdb.ipynb           # Notebook principal avec implémentation
│   └── figures/                 # Visualisations générées
├── report/                       # Rapport académique
│   ├── introduction.md          # Introduction du projet
│   └── final_discussion.md      # Discussion finale comparative
├── requirements.txt              # Dépendances Python
├── README.md                     # Documentation du projet
└── PROJECT_SUMMARY.md           # Ce fichier
```

## Contenu par Partie

### Partie I : MLP pour Données Tabulaires

**Dataset** : Breast Cancer Wisconsin (classification binaire)

**Théorie couverte** :
- Perceptron et MLP
- nn.Module vs nn.Sequential
- Paramètres, gradients, forward/backpropagation
- state_dict et CPU vs GPU
- Méthodes d'initialisation (Constante, Gaussienne, Xavier)

**Implémentation** :
- Version nn.Sequential
- Version classe personnalisée héritant de nn.Module
- Analyse des paramètres (named_parameters, state_dict)
- Comparaison de 3 méthodes d'initialisation
- Entraînement complet avec loss/accuracy curves
- Sauvegarde et rechargement du modèle

**Évaluation** :
- Accuracy, Precision, Recall, F1-score
- Matrice de confusion
- Courbe ROC et AUC
- Distribution des probabilités

**Résultats** :
- Accuracy : ~97%
- F1-Score : ~0.97
- AUC-ROC : ~0.99

**Question de synthèse** : "Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente pour la classification tabulaire sur un dataset réel, et quelles sont ses limites ?"

### Partie II : CNN pour Vision par Ordinateur

**Dataset** : Fashion-MNIST (classification d'images 10 classes)

**Théorie couverte** :
- Pourquoi MLP est mal adapté aux images
- Localité, partage des poids, réceptive field
- Padding, stride, pooling
- Convolution 1×1, LeNet
- Calculs manuels de convolution et pooling

**Implémentation** :
- Implémentation manuelle (corr2d, max pooling, average pooling)
- Comparaison avec nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d
- Construction d'un CNN inspiré de LeNet
- Calculs de dimensions

**Expérimentations** :
- Effet du padding
- Max Pooling vs Average Pooling
- Nombre de filtres
- Visualisation des feature maps et filtres appris

**Comparaison** :
- MLP vs CNN sur Fashion-MNIST
- Tableau comparatif des performances

**Résultats** :
- CNN Accuracy : ~91%
- MLP Accuracy : ~88%
- CNN Parameters : ~60,000
- MLP Parameters : ~530,000

**Question de synthèse** : "Pourquoi un CNN est-il plus pertinent qu'un MLP pour la classification d'images et comment les choix architecturaux influencent-ils les performances ?"

### Partie III : RNN – LSTM – GRU – Seq2Seq

**Dataset** : IMDb (analyse de sentiments - simulé pour démonstration)

**Théorie couverte** :
- Modèle de langage, règle de chaîne, probabilités conditionnelles
- Perplexité
- RNN, LSTM, GRU
- Seq2Seq, Teacher Forcing, Beam Search
- BLEU Score, BPTT, Gradient Clipping
- Bidirectional RNN, Attention Mechanism
- Embeddings, Padding et masquage

**Implémentation** :
- Tokenisation et construction du vocabulaire
- Création des DataLoaders avec padding
- Implémentation RNN simple
- Implémentation LSTM
- Implémentation GRU
- Construction d'un modèle Seq2Seq
- Implémentation Beam Search et Greedy Decoding

**Expérimentations** :
- Comparaison RNN vs LSTM vs GRU
- Étude du gradient clipping
- Comparaison des temps d'entraînement
- Analyse de la convergence

**Résultats** :
- LSTM et GRU surpassent significativement le RNN simple
- Gradient clipping améliore la stabilité
- GRU offre un bon compromis performance/complexité

**Question de synthèse** : "Dans quelle mesure les architectures récurrentes permettent-elles de modéliser efficacement une séquence réelle et pourquoi passer d'un RNN simple à un LSTM/GRU puis à Seq2Seq ?"

## Discussion Finale

La discussion finale compare systématiquement les trois architectures :

**Pourquoi le Deep Learning adapte ses architectures selon le type de données** :
- Structure inhérente des données (spatiale, temporelle, tabulaire)
- Induction bias de chaque architecture
- Efficacité computationnelle

**Comparaison MLP vs CNN vs RNN** :
- Tableau comparatif des caractéristiques
- Avantages et limites de chaque approche
- Domaines d'application optimaux

**Dépendances spatiales et temporelles** :
- Exploitation par les CNN (localité, partage de poids)
- Exploitation par les RNN (mémoire temporelle, état caché)
- Comparaison des approches

**Recommandations pratiques** :
- Choix de l'architecture selon le type de données
- Bonnes pratiques pour chaque famille
- Évolution vers les architectures modernes (Transformers)

## Installation et Exécution

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Exécution des notebooks

Lancer les notebooks Jupyter dans l'ordre :

1. `part1_mlp/mlp_breast_cancer.ipynb`
2. `part2_cnn/cnn_fashion_mnist.ipynb`
3. `part3_rnn/rnn_imdb.ipynb`

## Points Clés du Projet

### Partie I (MLP)
- L'initialisation des poids est critique (He/Kaiming pour ReLU)
- La régularisation (Dropout, L2) est essentielle pour éviter l'overfitting
- Le MLP atteint d'excellentes performances sur les données tabulaires (~97%)
- Limites : manque d'explicabilité, sensibilité aux hyperparamètres

### Partie II (CNN)
- Le CNN exploite la structure spatiale des images
- Le partage des poids réduit drastiquement le nombre de paramètres
- Le CNN surpasse le MLP pour les images avec moins de paramètres
- Les choix architecturaux (padding, pooling, filtres) influencent les performances

### Partie III (RNN)
- Le RNN simple souffre de vanishing/exploding gradients
- LSTM et GRU résolvent ces problèmes via des mécanismes de portes
- Le gradient clipping est essentiel pour la stabilité
- Seq2Seq avec attention permet des tâches de transformation de séquence

### Discussion Finale
- Chaque architecture est optimisée pour un type de données spécifique
- L'induction bias est crucial pour l'efficacité
- Les architectures modernes (Transformers) combinent les avantages
- Le choix de l'architecture dépend de la structure des données et des contraintes

## Contributions Académiques

Ce projet fournit :

1. **Une étude théorique complète** des trois familles d'architectures
2. **Des implémentations PyTorch reproductibles** avec code commenté
3. **Des expérimentations comparatives** systématiques
4. **Des analyses critiques** des résultats et limites
5. **Une synthèse comparative** des approches
6. **Des réponses argumentées** aux questions de synthèse

## Format Académique

Le contenu est rédigé dans un style académique prêt à être intégré dans un rapport universitaire ou un mémoire, avec :
- Citations et références théoriques
- Analyses scientifiques des résultats
- Interprétations critiques
- Tableaux comparatifs
- Visualisations professionnelles

## Limitations et Perspectives

**Limitations** :
- Dataset IMDb simulé (pour démonstration)
- Optimisation des hyperparamètres non exhaustive
- Ressources computationnelles limitées

**Perspectives** :
- Utilisation du vrai dataset IMDb
- Recherche systématique d'hyperparamètres
- Expérimentation avec des architectures plus profondes
- Comparaison avec des architectures modernes (Transformers)
- Extension à d'autres types de données (graphes, audio)

## Conclusion

Ce projet a démontré l'importance d'adapter l'architecture de Deep Learning au type de données pour maximiser les performances. Les résultats expérimentaux confirment que :
- Le MLP est excellent pour les données tabulaires
- Le CNN surpasse le MLP pour les images grâce à l'exploitation de la structure spatiale
- LSTM et GRU surpassent le RNN simple grâce à leurs mécanismes de contrôle

Le projet fournit une base solide pour comprendre les différentes architectures de Deep Learning et leurs domaines d'application, avec un contenu académique complet prêt à être intégré dans un rapport universitaire.
