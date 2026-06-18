# Introduction

## Contexte du Projet

Le Deep Learning a révolutionné de nombreux domaines en permettant l'apprentissage automatique de représentations complexes à partir de données brutes. Contrairement aux méthodes classiques de machine learning qui nécessitent souvent un feature engineering manuel, les réseaux de neurones profonds peuvent apprendre automatiquement des features hiérarchiques adaptées à la tâche.

Cependant, toutes les données ne se ressemblent pas. Les images ont une structure spatiale, le texte a une structure temporelle, et les données tabulaires ont des relations entre features. Pour exploiter efficacement ces différentes structures, le Deep Learning a développé des architectures spécialisées : les CNN (Convolutional Neural Networks) pour les images, les RNN (Recurrent Neural Networks) pour les séquences, et les MLP (Multi-Layer Perceptrons) pour les données tabulaires.

## Objectif du Projet

Ce projet a pour objectif de réaliser une étude comparative approfondie de trois familles d'architectures de Deep Learning :

1. **MLP (Multi-Layer Perceptron)** pour la classification de données tabulaires (Breast Cancer Wisconsin)
2. **CNN (Convolutional Neural Network)** pour la classification d'images (Fashion-MNIST)
3. **RNN/LSTM/GRU/Seq2Seq** pour le traitement de séquences textuelles (IMDb)

Pour chaque architecture, nous avons :
- Étudié la théorie sous-jacente
- Implémenté les modèles en PyTorch
- Réalisé des expérimentations comparatives
- Analysé les résultats de manière critique
- Répondu à une question de synthèse

## Structure du Rapport

Ce rapport est organisé en quatre parties principales :

**Partie I : MLP pour Données Tabulaires**
- Théorie des perceptrons et MLP
- Analyse exploratoire du dataset Breast Cancer Wisconsin
- Implémentation PyTorch (nn.Sequential et classe personnalisée)
- Comparaison des méthodes d'initialisation
- Entraînement et évaluation
- Analyse critique et réponse à la question de synthèse

**Partie II : CNN pour Vision par Ordinateur**
- Théorie des CNN et vision par ordinateur
- Calculs manuels de convolution et pooling
- Implémentation manuelle et comparaison avec PyTorch
- Construction d'un CNN inspiré de LeNet
- Étude expérimentale des hyperparamètres
- Visualisation des feature maps et filtres
- Comparaison MLP vs CNN
- Analyse critique et réponse à la question de synthèse

**Partie III : RNN – LSTM – GRU – Seq2Seq**
- Théorie des modèles de langage et architectures récurrentes
- Préparation des données IMDb (tokenisation, vocabulaire, padding)
- Implémentation RNN, LSTM, et GRU
- Comparaison expérimentale
- Étude du gradient clipping
- Construction d'un modèle Seq2Seq
- Implémentation Beam Search
- Analyse critique et réponse à la question de synthèse

**Discussion Finale**
- Pourquoi le Deep Learning adapte ses architectures selon le type de données
- Comparaison entre MLP, CNN et RNN
- Différences entre données tabulaires, images et séquences
- Dépendances spatiales et temporelles
- Avantages et limites de chaque famille de modèles
- Recommandations pratiques

## Contributions Scientifiques

Ce projet contribue à la compréhension des architectures de Deep Learning en :

1. **Démontrant empiriquement** l'importance d'adapter l'architecture au type de données
2. **Comparant systématiquement** les performances de différentes architectures sur leurs domaines d'application
3. **Analysant l'impact** des choix architecturaux (initialisation, padding, pooling, etc.)
4. **Fournissant une synthèse critique** des forces et faiblesses de chaque approche

## Méthodologie

Pour chaque partie, nous avons suivi une méthodologie rigoureuse :

1. **Étude théorique** : Compréhension des concepts fondamentaux
2. **Analyse exploratoire** : Étude des données et visualisations
3. **Prétraitement** : Nettoyage, normalisation, division train/validation/test
4. **Implémentation** : Code PyTorch complet et reproductible
5. **Expérimentation** : Comparaison de différentes configurations
6. **Évaluation** : Métriques standards (accuracy, precision, recall, F1-score)
7. **Analyse critique** : Interprétation des résultats et identification des limites

## Outils et Technologies

- **Langage** : Python
- **Framework** : PyTorch 2.0+
- **Bibliothèques** : NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Datasets** : Breast Cancer Wisconsin, Fashion-MNIST, IMDb
- **Environnement** : Jupyter Notebooks

## Organisation des Fichiers

```
deeplearning/
├── part1_mlp/
│   ├── theory_mlp.md
│   ├── mlp_breast_cancer.ipynb
│   └── figures/
├── part2_cnn/
│   ├── theory_cnn.md
│   ├── cnn_fashion_mnist.ipynb
│   └── figures/
├── part3_rnn/
│   ├── theory_rnn.md
│   ├── rnn_imdb.ipynb
│   └── figures/
├── report/
│   ├── introduction.md
│   ├── final_discussion.md
│   └── ...
├── requirements.txt
└── README.md
```

## Public Cible

Ce rapport s'adresse aux :
- Étudiants en informatique et data science souhaitant approfondir leurs connaissances en Deep Learning
- Chercheurs intéressés par une comparaison systématique des architectures de Deep Learning
- Practitioners cherchant à choisir l'architecture adaptée à leur type de données

## Limitations

Certaines limitations doivent être notées :

1. **Dataset simulé pour IMDb** : Pour des raisons de temps et de ressources, le dataset IMDb a été simulé. Dans un contexte réel, il faudrait utiliser le vrai dataset IMDb.
2. **Hyperparamètres** : L'optimisation des hyperparamètres n'a pas été exhaustive. Une recherche plus systématique pourrait améliorer les performances.
3. **Ressources computationnelles** : Certaines expériences (ex: très grands réseaux) n'ont pas été réalisées par manque de ressources GPU.

Malgré ces limitations, ce projet fournit une base solide pour comprendre les différentes architectures de Deep Learning et leurs domaines d'application.
