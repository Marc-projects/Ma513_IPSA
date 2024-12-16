# Ma513_IPSA

## Description
Ce projet à pour but d'entrainer des modèles de BERT capables de reconnaitre les ner-tags de tous les mots d'une phrase. Nous disposons, pour l'entrainement, trois fichiers. Une premier fichier de train, une deuxième de validation et un dernier de test.

1. **Entraînement des modèles BERT** : Utilisation de fichiers JSON contenant des données NER pour entraîner plusieurs modèles BERT.
2. **Prédiction multi-modèles** : Pour chaque modèle entraîné, les prédictions sont effectuées sur des données de validation et de test. Les résultats de plusieurs modèles sont ensuite croisés pour obtenir une prédiction finale.

Les données utilisées sont au format JSON, et les résultats sont générés sous forme de fichiers JSONLines contenant les prédictions.

## Fonctionnement

### CSV de gestion des modèles
Les modèles BERT à importer depuis internet qui seront entrainés et utilisés pour nos données sont définis dans le fichier CSV `models.csv`. Il est de la forme suivante : 

```csv
model_name
google-bert/bert-large-uncased
google-bert/bert-large-cased
google-bert/bert-base-cased
bert-base-uncased
jackaduma/SecBERT
```

### Entraînement des modèles
- **Lecture des fichiers JSON** : Les fichiers `NER-TRAINING.jsonlines` et `NER-VALIDATION.jsonlines` sont lus pour extraire les tokens et les labels associés.
- **Tokenisation** : Chaque phrase est tokenisée à l'aide de BERT et les NER-tags sont ajustés pour s'aligner avec les tokens créés sans que le padding influence les calculs de coûts.
- **Entraînement** : Les modèles BERT sont entraînés sur les données d'entraînement et évalués sur les données de validation. Les modèles et les tokenizers sont ensuite sauvegardés dans un répertoire dédié.

### Prédiction multi-modèles
- **Chargement des modèles pré-entraînés** : Les modèles sauvegardés sont rechargés pour effectuer des prédictions sur les données de validation et de test.
- **Prédictions** : Chaque modèle génère des prédictions pour les entités nommées sur les jeux de données de validation et de test. Les prédictions sont ensuite croisées pour obtenir la prédiction finale en choisissant le label le plus fréquent parmi les modèles.
- **Évaluation** : Les performances des modèles sont évaluées à l'aide de la fonction `classification_report` de scikit-learn sur les données de validation.
- **Sauvegarde des résultats** : Les résultats des prédictions croisées sont sauvegardés dans un fichier `NER-TESTING_bert_multi_pred1.jsonlines`.

## Exécution

1. **Modification du fichier `models.csv`**
Il suffit d'écrire dans le fichier `models.csv` les noms des modèles pré-entrainés de `huggingface.co` à utiliser.

2. **Execution du script `model_BERT.py`**
Ce script va automatiquement entrainer et enregistrer tous les modèles inscrits dans le fichier `models.csv`.

3. **Execution du script `predict_multi_BERT.py`**
Ce code utilise les modèles précédemment entrainés pour faire des prédictions sur les validations data et sur les tests data. Pour les données de validation, il affiche d'un part le résultat du croisement des modèles, et d'autre part les performances individuelles.