# Ma513_IPSA

1. **Entraînement des modèles BERT** : Utilisation de fichiers JSON contenant des données NER pour entraîner plusieurs modèles BERT.
2. **Prédiction multi-modèles** : Pour chaque modèle entraîné, les prédictions sont effectuées sur des données de validation et de test. Les résultats de plusieurs modèles sont ensuite croisés pour obtenir une prédiction finale.

Les données utilisées sont au format JSON, et les résultats sont générés sous forme de fichiers JSONLines contenant les prédictions.

## Fonctionnement

### Entraînement des modèles
- **Lecture des fichiers JSON** : Les fichiers `NER-TRAINING.jsonlines` et `NER-VALIDATION.jsonlines` sont lus pour extraire les tokens et les labels associés (ou les identifiants uniques pour les données de test).
- **Tokenisation** : Chaque phrase est tokenisée à l'aide de BERT et des tags NER sont ajustés pour s'aligner avec les tokens créés.
- **Entraînement** : Les modèles BERT sont entraînés sur les données d'entraînement et évalués sur les données de validation. Les modèles et les tokenizers sont ensuite sauvegardés dans un répertoire dédié.

### Prédiction multi-modèles
- **Chargement des modèles pré-entraînés** : Les modèles sauvegardés sont rechargés pour effectuer des prédictions sur les données de validation et de test.
- **Prédictions** : Chaque modèle génère des prédictions pour les entités nommées sur les jeux de données de validation et de test. Les prédictions sont ensuite croisées pour obtenir la prédiction finale en choisissant le label le plus fréquent parmi les modèles.
- **Évaluation** : Les performances des modèles sont évaluées à l'aide de la fonction `classification_report` de scikit-learn sur les données de validation.
- **Sauvegarde des résultats** : Les résultats des prédictions croisées sont sauvegardés dans un fichier `NER-TESTING_bert_multi_pred1.jsonlines`.

### CSV de gestion des modèles
Les modèles BERT sont définis dans un fichier CSV `models.csv`. Ce fichier contient les noms des modèles BERT à utiliser pour l'entraînement et la prédiction. Exemple de contenu du fichier `models.csv` :

```csv
model_name
google-bert/bert-large-uncased
google-bert/bert-large-cased
google-bert/bert-base-cased
bert-base-uncased
jackaduma/SecBERT
```
