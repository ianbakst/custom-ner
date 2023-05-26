"""train_model.py Main Training script for Named Entity Recognition
Ian Bakst, DataOps Engineer. Tamr, Inc.

Trains the model from a Json File of Labeled Data.
Example Format of the input json is:
{"content": "Word b&w headphones ",
 "annotation": [{"label": ["Brand"],
                 "points": [{"text": "b&w",
                             "start": 5,
                             "end": 7}]},
                {"label": ["Category"],
                 "points": [{"text": "headphones",
                             "start": 9,
                             "end": 18}]}]}
Each record is its own line and json object.
The entire file is NOT a list of json objects, i.e. no commas at the ends of lines.

"""
from __future__ import unicode_literals, print_function
import plac
import os
import logging
import random
from pathlib import Path
import json
import pickle
import spacy
from spacy.util import minibatch, compounding
from spacy.tokenizer import Tokenizer


# Input options and their Flags
@plac.annotations(input_file=("Input file", "option", "i", str),
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
    doccano=("Is the Input from Doccano","flag","d"))


def main(input_file=None,model=None, new_model_name='new_model', output_dir=None, n_iter=10,doccano=False):
    V = {'input_file': input_file,
            'model': model,
            'new_model_name': new_model_name,
            'output_dir': output_dir,
            'n_iter': n_iter}
    if doccano==True:
        fname = from_doccano(V['input_file'])
        convert_file(fname)
    else:
        convert_file(input_file=V['input_file'])
    train_model(model=V['model'],
                new_model_name=V['new_model_name'],
                output_dir=V['output_dir'],
                n_iter=V['n_iter'])


def from_doccano(infile, outfile = None):
    if outfile:
        fname = outfile
    else:
        fname = infile + '.converted'
    f = open(infile,'r')
    with open(fname, "w") as o:
        for line in f:
            entry = json.loads(line)
            ann = [{"label": [k[2]],"points": [{"text":entry['text'][k[0]:k[1]],"start":k[0],"end":k[1]-1}]} for k in entry['labels']]
            new = {"content": entry['text'],
                  "annotation": ann}
            o.write(str(json.dumps(new)))
            o.write('\n')
    f.close()
    return fname


def create_custom_tokenizer(nlp,):
    my_prefix = r'[0-9]\.'

    all_prefixes_re = spacy.util.compile_prefix_regex(tuple(list(nlp.Defaults.prefixes) + [my_prefix]))

    # Handle ( that doesn't have proper spacing around it
    custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()]']
    infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))

    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, nlp.Defaults.tokenizer_exceptions,
                     prefix_search=all_prefixes_re.search,
                     infix_finditer=infix_re.finditer, suffix_search=suffix_re.search,
                     token_match=None)


def convert_file(input_file,dir='data/processed',write_file=True):
    try:
        training_data = []
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))

            training_data.append((text, {"entities" : entities}))
        if write_file:
            with open(os.path.join(dir,'spacy_input'), 'wb') as fp:
                pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        training_data = None
    finally:
        return training_data


def my_tokenizer_pri(nlp,my_prefix,my_infix,my_suffix=[]):
    all_prefixes = spacy.util.compile_prefix_regex(tuple(list(nlp.Defaults.prefixes) + my_prefix))
    all_infixes = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + my_infix))
    all_suffixes = spacy.util.compile_suffix_regex(tuple(list(nlp.Defaults.suffixes) + my_suffix))
    return Tokenizer(nlp.vocab,nlp.Defaults.tokenizer_exceptions,
                     prefix_search = all_prefixes.search,
                     suffix_search = all_suffixes.search,
                     infix_finditer = all_infixes.finditer,
                     token_match = None)


def train_model(model=None, new_model_name='new_model', output_dir=None, n_iter=10,input_dir='data/processed'):

    with open(os.path.join(input_dir,'spacy_input'), 'rb') as fp:
        TRAIN_DATA = pickle.load(fp)

    LABEL = set([item for sublist in [[r[2] for r in R[1]['entities']] for R in TRAIN_DATA] for item in sublist])
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    my_prefix = [r'''^[±\-\+0-9., ]+[0-9 ]+''']
    my_infix = [r'''[/]''']
    nlp.tokenizer = my_tokenizer_pri(nlp, my_prefix, my_infix)

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
        print('Losses:',losses)

    # Save model
    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


if __name__ == '__main__':
    plac.call(main)
