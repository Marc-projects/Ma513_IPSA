# Ma513_IPSA

## Description
This project aims to train BERT models capable of recognizing the NER tags of all words in a sentence. For training, we have three files: a training file, a validation file, and a test file.

1. **Training BERT models** : Using JSON files containing NER data to train several BERT models.
2. **Multi-model prediction** : For each trained model, predictions are made on validation and test data. The results from multiple models are then combined to obtain a final prediction.

The data used are in JSONLines format, and the results are generated as JSONLines files containing the predictions.

## Functioning

### Model management CSV
The BERT models to be imported from the internet, trained, and used for our data are defined in the CSV file `models.csv`. It has the following format:

```csv
model_name
google-bert/bert-large-uncased
google-bert/bert-large-cased
google-bert/bert-base-cased
bert-base-uncased
jackaduma/SecBERT
```

### Training the models
- **Reading JSON files** : The files `NER-TRAINING.jsonlines` and `NER-VALIDATION.jsonlines` are read to extract the tokens and their associated labels.
- **Tokenization** : Each sentence is tokenized using BERT, and the NER tags are adjusted to align with the created tokens without padding influencing the cost calculations.
- **Training** : BERT models are trained on the training data and evaluated on the validation data. The models and tokenizers are then saved in a dedicated directory.

### Multi-model prediction
- **Loading pre-trained models** : The saved models are reloaded to make predictions on the validation and test data.
- **Predictions** : Each model generates predictions for named entities on the validation and test datasets. The predictions are then combined to obtain the final prediction by choosing the most frequent label among the models.
- **Evaluation** : The models' performances are evaluated using the `classification_report` function from scikit-learn on the validation data.
- **Saving results** : The results of the combined predictions are saved in a file named `NER-TESTING_bert_multi_pred1.jsonlines`.

## Execution

1. **Modify the `models.csv` file** : 
Simply write the names of the pre-trained models from `huggingface.co` to be used in the `models.csv` file.

2. **Run the `model_BERT.py` script** : 
This script will automatically train and save all the models listed in the `models.csv` file.

3. **Run the `predict_multi_BERT.py` script** : 
This code uses the previously trained models to make predictions on the validation and test data. For the validation data, it displays both the combined model results and the individual performances.