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
        mot = mot.replace(str(key), str(value))
    return mot

dic_tag = {'O' : 0, 'B-Entity' : 1, 'B-Action' : 2, 'I-Action' : 3, 'I-Entity' : 4, 'B-Modifier' : 5, 'I-Modifier' : 6}
dic_tag_inv = {0 : 'O', 1 : 'B-Entity', 2 : 'B-Action', 3 : 'I-Action', 4 : 'I-Entity', 5 : 'B-Modifier', 6 : 'I-Modifier'}

def prepa_data_pred(file, tokenizer, labels = False):
    corpus = read_json(file)
    if labels:
        data = {"tokens": [[t if t != "\uf0b7" else "/uf0b7" for t in val["tokens"]] for val in corpus], "ner_tags": [val["ner_tags"] for val in corpus]}   # récupération des phrases et de leurs NER tags en remplacant les caractères problématiques
    else:
        data = {"tokens": [[t if t != "\uf0b7" else "/uf0b7" for t in val["tokens"]] for val in corpus], "unique_id": [val["unique_id"] for val in corpus]}   # récupération des phrases et de leurs NER tags en remplacant les caractères problématiques
    tokenized_inputs = tokenizer(data['tokens'], truncation=True, padding=True, is_split_into_words=True)
    isLabels = []
    for i in range(len(data["tokens"])):
        add = -1
        isLabel = []
        for j in tokenized_inputs.word_ids(i):
            if (j != None) and (add != j):
                isLabel.append(True)    # création d'une liste pour savoir s'il faut récupérer la prediction faites 
                add = j
            else:
                isLabel.append(False)
        isLabels.append(isLabel)
    if labels:
        return tokenized_inputs, isLabels, data['ner_tags']
    else:
        return tokenized_inputs, isLabels, data

def prediction_finale(model, data, islabel):
    with torch.no_grad():
        outputs = model(**{key: torch.tensor(val) for key, val in data.items()})
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # prédiction avec pytorch

    pred_words = []
    for i in range(len(predictions)):
        pred = []
        for j in range(len(predictions[i])):
            if islabel[i][j]:
                pred.append(replace(str(int(predictions[i][j])), dic_tag_inv))  # récupération des predictions et conversion des tags en texte
        pred_words.append(pred)
    return pred_words

def multi_prediction_finale(models_saves_names):
    multi_pred_val = []
    multi_pred_test = []
    for model_save in models_saves_names:

        model_reloaded = BertForTokenClassification.from_pretrained(model_save)
        tokenizer_reloaded = BertTokenizerFast.from_pretrained(model_save)

        tokenize_val, isLabels_val, ner_tags_val = prepa_data_pred("data/NER-VALIDATION.jsonlines", tokenizer_reloaded, labels=True)
        tokenize_test, isLabels_test, data_test = prepa_data_pred("data/NER-TESTING.jsonlines", tokenizer_reloaded, labels=False)

        pred_val_final = prediction_finale(model_reloaded, tokenize_val, isLabels_val)
        pred_test_final = prediction_finale(model_reloaded, tokenize_test, isLabels_test)

        pred_val_flat = [item for sublist in pred_val_final for item in sublist]
        ner_tags_val_flat = [item for sublist in ner_tags_val for item in sublist]
        print(model_save)
        print(classification_report(ner_tags_val_flat, pred_val_flat))   # affichage des performances individuelles

        multi_pred_val.append(pred_val_final)   # récupération des predictions
        multi_pred_test.append(pred_test_final) # récupération des predictions
    return multi_pred_val, multi_pred_test, ner_tags_val, data_test

def val_freq_high(liste):
    compteur={}
    for i in liste:
        if i in compteur.keys():
            compteur[i] += 1
        else:
            compteur[i] = 1
    high = max(compteur, key=compteur.get)  # retourne la clé avec la plus grande valeur
    return high

def mean_multi_prediction_finale(multi_pred):
    pred_croise = []
    for i in range(len(multi_pred[0])):
        label = []
        for j in range(len(multi_pred[0][i])):
            label.append([multi_pred[0][i][j]])
        pred_croise.append(label)
    for n in range(1, len(multi_pred)):
        for i in range(len(multi_pred[n])):
            for j in range(len(multi_pred[n][i])):
                pred_croise[i][j].append(multi_pred[n][i][j])
    for i in range(len(pred_croise)):
        for j in range(len(pred_croise[i])):
            pred_croise[i][j] = val_freq_high(pred_croise[i][j])    # récupération de la prédiction avec le plus de fréquence
    return pred_croise
    

models_saves_names = ["./save_google-bert/bert-large-uncased", "./SEC_BERT_50_epochs", "./save_bert-base-uncased", "./save_google-bert/bert-base-cased", "./save_jackaduma/SecBERT"]    # liste des modeles pré-entrainé et adapaté à nos besoins

multi_pred_val, multi_pred_test, ner_tags_val, data_test = multi_prediction_finale(models_saves_names)

pred_croise_val = mean_multi_prediction_finale(multi_pred_val)
pred_croise_test = mean_multi_prediction_finale(multi_pred_test)


pred_croise_val_flat = [item for sublist in pred_croise_val for item in sublist]
ner_tags_val_flat = [item for sublist in ner_tags_val for item in sublist]
print("prédiction croisée")
print(classification_report(ner_tags_val_flat, pred_croise_val_flat))


with open("NER-TESTING_bert_multi_pred1.jsonlines", "w") as json_file:
    pass

for i in range(len(pred_croise_test)):
    with open("NER-TESTING_bert_multi_pred1.jsonlines", "a") as json_file:
        if len(data_test["tokens"][i]) == len(pred_croise_test[i]):
            json_file.write(json.dumps({"unique_id" : data_test["unique_id"][i], "tokens" : [t if t != "/uf0b7" else "\uf0b7" for t in data_test["tokens"][i]], "ner_tags" : pred_croise_test[i]})+"\n")
        else:
            print(f"nombre de tags differents du nombre de mots à la ligne {i}")

