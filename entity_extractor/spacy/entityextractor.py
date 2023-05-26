import os, shutil
import sys
import yaml
import logging
import argparse
import json
import spacy
import time
import warnings
import random
import re
from spacy.lang.en import English
from spacy.util import minibatch, compounding
from entity_extractor.utils import ee_dir

##### Append repo location to path
sys.path.append(".")

from entity_extractor.spacy import spacy_scripts
from entity_extractor.utils import setup_logging, ask_user


with open(os.path.join(ee_dir,'conf','config.yaml')) as config_file:
    config = yaml.load(config_file,Loader = yaml.FullLoader)

def list_all_models(mod_dir):
    """Lists all model metadata.

    """
    data = []
    models = [i for i in os.walk(os.path.join(config['workspace_home'], mod_dir))][0][1]

    for model in models:
        with open(os.path.join(config['workspace_home'], mod_dir, model, 'meta.json'),'r') as f:
            data.append(json.load(f))

    return data

def delete_all_models(mod_dir):
    """Delete all models.

    """
    current_list = list_all_models(mod_dir)

    for entry in current_list:
        delete_model(entry['name'])

    return

def delete_model(mname, mod_dir):
    """Delete model {mname}.

    """

    shutil.rmtree(os.path.join(config["workspace_home"], mod_dir, mname))

    return

def create_model(mod_dir, name, lang = 'en', pipes = ['ner']):

    current_list = list_all_models(mod_dir)

    all_models = [cd['name'] for cd in current_list]

    if name in all_models:
        return f"Model '{name}' already exists", 409

    nlp = spacy.blank(lang)

    for p in pipes:
        i = nlp.create_pipe(p)
        nlp.add_pipe(i)

    nlp.meta['name'] = name
    nlp.meta['last_trained'] = f'INITIALIZED:{time.strftime("%Y%m%d-%H%M%S")}'

    nlp.begin_training()
    nlp.to_disk(os.path.join(config['workspace_home'], config['model_dir'], name))

    return nlp, 201


def initialize_model(config, model = None):

    datestring = time.strftime("%Y%m%d-%H%M%S")

    if model == None:
        logging.info(f'Model name is empty. Creating a blank Spacy English model.')
        nlp = create_model(config['mod_dir'], config['untrained_model_name'])

    else:
        logging.info(f'Loading Spacy model {model}.')
        nlp = spacy.load(os.path.join(config["workspace_home"], config['model_dir'], model))
        logging.info(f'Spacy model {model} loaded. Model pipeline: {nlp.pipe_names}.')

    return nlp

def update_tokenizer(config, nlp, prefix = config['tokenizer_prefixes'],
                                  infix = config['tokenizer_infixes'],
                                  suffix = config['tokenizer_suffixes']):

    if any(prefix, infix, suffix):
        nlp.tokenizer = spacy_scripts.extend_tokenizer(nlp, prefix, infix, suffix)

    output_dir = os.path.join(config["workspace_home"], config["model_dir"], nlp.meta["name"])

    nlp.to_disk(output_dir)

    return nlp

def train_model(config, train_data, nlp, n_iter = 10):

    if 'ner' not in nlp.pipe_names:
        logging.info(f'Adding Named Entity Recognizer (NER) to model pipeline.')
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        logging.info(f'Retrieving Named Entity Recognizer (NER) from model pipeline.')
        ner = nlp.get_pipe('ner')

    logging.info(f'Extracting set of labels from training data.')
    labels = spacy_scripts.get_label_set(train_data)

    logging.info(f'The following labels will be added to the NER: {labels}.')
    ner = spacy_scripts.append_labels(ner, labels)

    if config["untrained_model_name"] in nlp.meta["name"]:
        logging.info(f'Starting new model training:'
                       f'\nModel pipeline: {nlp.pipe_names}.'
                       f'\nTrained components: {config["trained_pipe_components"]}'
                       f'\nNER labels: {ner.labels}.')
        optimizer = nlp.begin_training()
    else:
        logging.info(f'Starting existing model training:'
                       f'\nModel pipeline: {nlp.pipe_names}.'
                       f'\nTrained components: {config["trained_pipe_components"]}'
                       f'\nNER labels: {ner.labels}.')
        optimizer = nlp.resume_training()

    # Get model pipeline components that are fixed during training
    fixed_pipe = [pipe for pipe in nlp.pipe_names if pipe not in config['trained_pipe_components']]

    if fixed_pipe:
        logging.warning(f'The following model pipeline components will not be trained: {fixed_pipe}.')

    with nlp.disable_pipes(*fixed_pipe), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category = UserWarning, module = 'spacy')

        start = time.time()

        for itn in range(n_iter):
            logging.info(f'Training epoch {itn + 1} out of {n_iter}.')
            random.shuffle(train_data)
            batches = minibatch(train_data, size = compounding(4., 32., 1.001))
            losses = {}

            count = 1

            for batch in batches:
                if (count % 10 == 0) or count == 1:
                    logging.info(f'Processing training data batch {count}.')
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd = optimizer, drop = 0.35,
                           losses = losses)
                count += 1

        end = time.time()
        logging.info(f'Training completed. Elapsed time: {round((end - start)/60)} minutes.')
        logging.info(f'Training losses: {losses}')

        output_dir = os.path.join(config["workspace_home"], config["model_dir"], nlp.meta["name"])

        logging.info(f'Saving model to {output_dir}.')

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        nlp.meta['last_trained'] = time.strftime("%d-%b-%Y %H:%M:%S")
        nlp.meta['losses'] = losses

        nlp.to_disk(output_dir)

        return nlp

def test_model(config, test_data, nlp, plot_cm = False):

    logging.info(f'Generating confusion matrix for model {nlp.meta["name"]}.')
    confusion_matrix = spacy_scripts.generate_confusion_matrix(test_data, nlp, normalize = False)
    confusion_matrix_norm = spacy_scripts.generate_confusion_matrix(test_data, nlp, normalize = True)

    logging.info(f'Getting test labels from {test_data}.')
    classes = spacy_scripts.get_label_set(test_data)
    print('test_model:',classes)

    if plot_cm:
        output_cm = os.path.join(config["workspace_home"], config["model_dir"], nlp.meta["name"], 'confusion_matrix.pdf')
        spacy_scripts.plot_confusion_matrix(nlp.meta['name'], confusion_matrix, classes, save_file = output_cm)

        logging.info(f'Confusion matrix for model {nlp.meta["name"]} saved to {output_cm}.')

        output_cm_norm = os.path.join(config["workspace_home"], config["model_dir"], nlp.meta["name"], 'confusion_matrix_norm.pdf')
        spacy_scripts.plot_confusion_matrix(nlp.meta['name'], confusion_matrix_norm, classes, save_file = output_cm_norm, normalize = True)

        logging.info(f'Confusion matrix (normalized) for model {nlp.meta["name"]} saved to {output_cm_norm}.')


    logging.info(f'Calculating model performance metrics for model {nlp.meta["name"]}.')
    nlp.meta['metrics'] = spacy_scripts.get_model_metrics(confusion_matrix, classes)

    return nlp

def predict_labels(nlp, data):

    docs = [{'recordId': d['recordId'],
             'nlp': nlp(d['description'])} for d in data]

    enriched_docs = [{'recordId': doc['recordId'],
                      'Description': doc['nlp'].text,
                      'Labels': [{ent.label_: ent.text} for ent in doc['nlp'].ents]} for doc in docs]

    return enriched_docs

if __name__ == '__main__':

    with open('analytics_solutions/EntityExtractor/config.yaml') as config_file:
        config = yaml.load(config_file,Loader = yaml.FullLoader)

    #setup_logging()

    parser = argparse.ArgumentParser(description = 'Trains a spancy NER model from a json file of labeled data.', add_help=True)
    parser.add_argument('train_file', action = "store", help = 'Training data (json).')
    parser.add_argument('-d', '--doccano', action = "store_true", help = 'Flag if train_file is in a Doccano json format.')
    parser.add_argument('-m', '--spacy_model', action = "store", default = None, type = str, metavar = '', help = 'Spacy model name.')

    args = parser.parse_args()
    train_file =  args.train_file
    doccano = args.doccano
    spacy_model =  args.spacy_model

    if not os.path.exists(train_file):
        logging.error(f'Training data file {train_file} not found. Exiting.')
        exit()

    if doccano:
        logging.info(f'Training data file {train_file} is in a Doccano json format. Converting to Spacy json format.')
        train_file = spacy_scripts.convert_doccano_json(train_file)

    logging.info(f'Reading training data from {train_file}.')
    train_data = spacy_scripts.read_json_to_list(train_file)

    ##### Perform train-test split & replace train_data with all_data above this.

    logging.info(f'Initializing model.')
    model = initialize_model(config, model = config['pretrained_model_name'])

    logging.info('Updating model tokenizer.')
    model = update_tokenizer(config, model)

    if ask_user(f'Train model {model.meta["name"]}?'):
        model = train_model(config, train_data, model, n_iter = config['training_iterations'])

    if ask_user(f'Test model {model.meta["name"]}?'):
        model_metrics, confusion_matrix = test_model(train_data, model) ## replace with test data

        if ask_user(f'Plot confusion matrix for {model.meta["name"]}?'):
            logging.info(f'Plotting confusion matrix for model {model.meta["name"]}.')
            spacy_scripts.plot_confusion_matrix(model.meta["name"], confusion_matrix, classes)
