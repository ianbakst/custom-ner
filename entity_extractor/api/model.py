import os, shutil
import random
import json
import spacy
from spacy.util import minibatch, compounding
from spacy.tokenizer import Tokenizer
import datetime
from collections import defaultdict
import itertools
import yaml

from entity_extractor.spacy import entityextractor as ee
from entity_extractor.spacy import spacy_scripts
from entity_extractor.utils import ee_dir

with open(os.path.join(ee_dir,'conf','config.yaml')) as config_file:
    config = yaml.load(config_file,Loader = yaml.FullLoader)

def list_all():
    """Lists all model metadata.

    """
    return ee.list_all_models(config['model_dir'])


def delete_all():
    """Delete all models.

    """

    return ee.delete_all_models(config['model_dir'])


def delete(mname):
    """Delete model {mname}.

    """

    return ee.delete_model(mname, config['model_dir'])


def create(body):
    """Create model {body.get('name')}.

    """

    try:
        name = body.get('name')
    except KeyError:
        return 'Model name not provided', 400

    lang = body.get('language', 'en')
    pipes = body.get('pipeline', ['ner'])

    nlp, flag = ee.create_model(config['model_dir'], name, lang, pipes)

    if flag == 409:
        return f"Model '{name}' already exists", 409
    else:
        return nlp.meta, 201


def update_tokenizer(mname, fixes):
    """Update tokenier for model {mname}.

    """

    try:
        nlp = spacy.load(os.path.join(config['workspace_home'], config['model_dir'], mname))
    except:
        return f"Model '{mname}' not found.", 404

    prefix = fixes.get('prefix',[])
    if not isinstance(prefix, list):
        prefix = [prefix]

    infix = fixes.get('infix',[])
    if not isinstance(infix, list):
        infix = [infix]

    suffix = fixes.get('suffix',[])
    if not isinstance(suffix, list):
        suffix = [suffix]

    nlp = ee.update_tokenizer(config, nlp, prefix, infix, suffix)

    return nlp.meta, 201


def train(mname, params):
    """Create model {mname}.

    """

    n = params.get('iterations', 10)
    dname = params.get('trainDataName')

    train_data = spacy_scripts.read_json_to_list(os.path.join(config['workspace_home'],
                                                 config['data_dir'], dname))

    try:
        nlp = spacy.load(os.path.join(config['workspace_home'], config['model_dir'], mname))
    except:
        return f"Model '{mname}' not found.", 404

    nlp = ee.train_model(config, train_data, nlp, n)

    return nlp.meta, 201


def test(mname, params):
    """Test model {mname}.

    """

    dname = params.get('testDataName')

    test_data = spacy_scripts.read_json_to_list(os.path.join(config['workspace_home'],
                                                 config['data_dir'], dname))

    try:
        nlp = spacy.load(os.path.join(config['workspace_home'], config['model_dir'], mname))
    except:
        return f"Model '{mname}' not found.", 404

    nlp = ee.test_model(config, test_data, nlp, plot_cm = True)

    return nlp.meta, 201


def predict(mname, data):
    """Generate predictions using model {mname}.

    """

    try:
        nlp = spacy.load(os.path.join(config['model_dir'], mname))
    except:
        return f"Model '{mname}' not found.", 404

    enriched_docs = ee.predict_labels(nlp, data)

    return enriched_docs, 200

def scores(mname,data):
    try:
        nlp = spacy.load(os.path.join(config['model_dir'],mname))
    except:
        return f"Model '{mname}' not found.", 404
    text_scores = []
    for text in data:
        # Build score Data
        score_data = []
        with nlp.disable_pipes('ner'):
            doc = nlp(text)
        beams = nlp.entity.beam_parse([doc], beam_width=16, beam_density=0.0001)
        entity_scores = defaultdict(float)
        for beam in beams:
            for score, ents in nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[(start, end, label)] += score
        for key in entity_scores:
            start, end, label = key
            score = entity_scores[key]
            score_data.append({'word': doc[start:end],
                               'label': [label, score]})
        score_data = sorted(score_data, key = keyword)
        for word, group in itertools.groupby(score_data, keyword):
            word_data = {}
            word_data["text"] = text.strip()
            word_data["word"] = word[0]
            for g in group:
                word_data[g['label'][0] + "__score"] = g['label'][1]
            text_scores.append(word_data)
    return text_scores, 200

def keyword(x):
    return x['word']
