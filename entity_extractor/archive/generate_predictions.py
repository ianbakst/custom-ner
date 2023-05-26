import spacy
import plac
import pandas as pd
from collections import defaultdict
import itertools


@plac.annotations(input_file=("Input File", "option", "i", str),
                  output_file=("Output File Name", "option","o",str),
                  model=("Trained model to Load","option","m",str))


def main(input_file=None,output_file=None,model=None):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except:
        print('Unable to read file. Aborting.')
        return

    try:
        nlp = spacy.load(model)
    except:
        print('Unable to load model. Aborting.')

    texts = [line[:-1] for line in lines]
    df = pd.DataFrame()
    text_scores = []
    for text in texts:
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
        score_data = sorted(score_data, key=keyword)
        for word, group in itertools.groupby(score_data, keyword):
            word_data = {}
            word_data["text"] = text.strip()
            word_data["word"] = word[0]
            for g in group:
                word_data[g['label'][0] + "__score"] = g['label'][1]
            text_scores.append(word_data)
        doc = nlp(text)
        # Build Entity Extractor
        d = {'Description':doc.text}
        for ent in doc.ents:
            d[ent.label_] = ent.text
        df = df.append(d,ignore_index=True)
    df.to_csv(output_file,index=False)
    df_scores = pd.DataFrame(text_scores)
    df_scores.to_csv(output_file+'.scores',index=False)


def keyword(x):
    return x['word']


if __name__ == '__main__':
    plac.call(main)