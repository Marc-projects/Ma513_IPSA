import json
import pandas as pd
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification, Trainer, TrainingArguments
import torch

def read_json(file):
    with open(file, 'r') as f:
        corpus_json = [json.loads(l) for l in list(f)]
    return corpus_json
    
def replace(mot, dic):
    for key, value in dic.items():
        mot = mot.replace(key, str(value))  # remplacement des mots en utilisant une paire clé valeur d'un dictionnaire
    return mot

dic_tag = {'O' : 0, 'B-Entity' : 1, 'B-Action' : 2, 'I-Action' : 3, 'I-Entity' : 4, 'B-Modifier' : 5, 'I-Modifier' : 6}
dic_tag_inv = {0 : 'O', 1 : 'B-Entity', 2 : 'B-Action', 3 : 'I-Action', 4 : 'I-Entity', 5 : 'B-Modifier', 6 : 'I-Modifier'}

def prepa_data(file, tokenizer):
    corpus = read_json(file)
    data = {"tokens": [[t if t != "\uf0b7" else "/uf0b7" for t in val["tokens"]] for val in corpus], "ner_tags": [val["ner_tags"] for val in corpus]}   # récupération des phrases et de leurs NER tags en remplacant les caractères problématiques
    tokenized_inputs = tokenizer(data['tokens'], truncation=True, padding=True, is_split_into_words=True)   # Véctorisation des phrases avec le tokenizer importé en ajoutant du padding et en précisant que les mots sont déjà séparés
    Labels = []
    for i in range(len(data["ner_tags"])):
        add = -1    # initialisation de la valeur qui permet de savoir si un label a déjà été importé 
        Label = []
        for j in tokenized_inputs.word_ids(i):
            if (j != None) and (add != j):  # condition pour savoir si le token d'une phrase est du padding ou si c'est le reste d'un mot séparé par le tokenizer
                Label.append(int(replace(data["ner_tags"][i][j], dic_tag)))
                add = j
            else:
                Label.append(-100)  # la valeur -100 permet de ne pas influencer les calculs de perte avec les tokens ajoutés dans la phrase par le tokenizer
        
        Labels.append(Label)

    tokenized_inputs["labels"] = Labels
    return Dataset.from_dict(tokenized_inputs)

def create_model(models_names):
    for model_name in models_names:
        model = BertForTokenClassification.from_pretrained(model_name, num_labels=7)    # import du modèle pré_entrainé
        tokenizer = BertTokenizerFast.from_pretrained(model_name)    # import du tokenizer pré_entrainé

        data_train = prepa_data("data/NER-TRAINING.jsonlines", tokenizer)
        data_val = prepa_data("data/NER-VALIDATION.jsonlines", tokenizer)

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=32,
            num_train_epochs=40,
            weight_decay=0.01,
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_val
        )

        trainer.train()

        model.save_pretrained('./save_'+model_name) # sauvegarde du modèle

        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        tokenizer.save_pretrained('./save_'+model_name) # sauvegarde du tokenizer


models_names = ["bert-base-uncased", "google-bert/bert-large-uncased", "google-bert/bert-base-cased", "google-bert/bert-large-cased"]   # liste des modèles à créer

create_model(models_names)