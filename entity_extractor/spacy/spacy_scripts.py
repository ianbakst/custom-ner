import logging
import json
import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spacy.tokenizer import Tokenizer
from spacy.gold import biluo_tags_from_offsets
from sklearn.metrics import confusion_matrix

plt.switch_backend('Agg')

################################################################################
# Scripts for input data processing

def convert_doccano_json(filename):

    try:
        f = open(filename,'r')

        output_filename = filename.rsplit( ".", 1 )[ 0 ] + '_spacy' + '.json'
        logging.info(f'Spacy JSON will be saved to {output_filename}.')

        with open(output_filename, "w") as output:
            for line in f:
                labels = []
                entry = json.loads(line)

                for label in entry['labels']:
                    labels.append(tuple(label))

                new_entry = (entry['text'], {'entities': labels})
                output.write(str(new_entry))
                output.write('\n')

        f.close()
        logging.info(f'{filename} conversion complete.')

    except Exception as e:
        logging.exception(f'Unable to process {input_file}.\nError: {e}')
        output_filename = None

    finally:
        return output_filename

def read_json_to_list(filename):

    data = []
    with open(filename, 'rb') as f:
        for line in f.readlines():
            data.append(eval(line))

    return data

################################################################################
# Scripts for Spacy model setup

def extend_tokenizer(nlp, prefix = [], infix = [], suffix = []):

    all_prefixes = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes + tuple(prefix))
    all_infixes = spacy.util.compile_infix_regex(nlp.Defaults.infixes + tuple(infix))
    all_suffixes = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes + tuple(suffix))

    return Tokenizer(nlp.vocab,nlp.Defaults.tokenizer_exceptions,
                     prefix_search = all_prefixes.search,
                     suffix_search = all_suffixes.search,
                     infix_finditer = all_infixes.finditer,
                     token_match = None)

def test_tokenizer(model, text):
    return [token.text for token in model.tokenizer(text)]

def get_label_set(dataset):
    labels = sorted(set([entity[2] for doc in dataset for entity in doc[1]['entities']]))
    labels.append('0')
    return labels

def append_labels(ner, labels):

    for label in labels:
        ner.add_label(label)

    return ner

################################################################################
# Scripts for Spacy model testing

def get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label

def create_target_vector(doc,nlp):
    text = nlp(doc[0])
    entities = doc[1]['entities']
    biluo_entities = biluo_tags_from_offsets(text, entities)
    return [get_cleaned_label(l) for l in biluo_entities]

def create_total_target_vector(docs,nlp):
    target_vector = []
    for doc in docs:
        target_vector.extend(create_target_vector(doc,nlp))
    return target_vector

def create_total_prediction_vector(docs, nlp):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc[0], nlp))
    return prediction_vector

def get_all_ner_predictions(text):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = biluo_tags_from_offsets(doc, entities)
    return bilou_entities

def create_prediction_vector(text, nlp):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text, nlp)]

def get_all_ner_predictions(text,nlp):

    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = biluo_tags_from_offsets(doc, entities)

    return bilou_entities

def generate_confusion_matrix(docs, nlp, normalize = False):

    classes = get_label_set(docs)
    y_true = create_total_target_vector(docs, nlp)
    y_pred = create_total_prediction_vector(docs, nlp)

    cm = confusion_matrix(y_true, y_pred, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm

def get_model_metrics(confusion_matrix, classes):

    TP = [confusion_matrix[i, i] for i in range(confusion_matrix.shape[0])]
    FP = [sum(confusion_matrix[:, i]) - confusion_matrix[i, i] for i in range(confusion_matrix.shape[0])]
    FN = [sum(confusion_matrix[i, :]) - confusion_matrix[i, i] for i in range(confusion_matrix.shape[0])]
    prec = [TP[i] / (TP[i] + FP[i]) for i in range(len(TP))]
    prec = [0.0 if x != x else x for x in prec]
    rec = [TP[i] / (TP[i] + FN[i]) for i in range(len(TP))]
    acc = sum(TP) / sum([sum(confusion_matrix[:,i]) for i in range(confusion_matrix.shape[0])])

    label_metrics = [{'label': classes[i],
                      'metrics': {'precision': prec[i],
                                  'recall': rec[i]}} for i in range(len(classes))]
    metrics = {'accuracy': acc,
               'labels': label_metrics}

    return metrics

def plot_confusion_matrix(model_name, confusion_matrix, classes, save_file = None, normalize = False):

    plt.subplots(figsize=(12,8))
    if normalize:
        ax = sns.heatmap(confusion_matrix, xticklabels = classes ,yticklabels = classes, cmap = "RdBu", linewidths=.0, annot=True, fmt = '.2f', cbar_kws = dict(ticks= np.linspace(0, np.max(confusion_matrix), num = 7, dtype = int)))
    else:
        ax = sns.heatmap(confusion_matrix, xticklabels = classes ,yticklabels = classes, cmap = "RdBu", linewidths=.0, annot=True, fmt = 'd', cbar_kws = dict(ticks= np.linspace(0, np.max(confusion_matrix), num = 7, dtype = int)))
    ax.set_xticks(np.arange(confusion_matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(confusion_matrix.shape[0]) + 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='center', fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center', fontsize = 10)

    ax.set_title(f'Confusion Matrix for Spacy NER Model {model_name}', fontsize = 12)

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, transparent = True, bbox_inches = 'tight')

    return
