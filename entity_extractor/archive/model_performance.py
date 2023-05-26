import plac
import spacy
from spacy.gold import biluo_tags_from_offsets
from sklearn.metrics import confusion_matrix
import json
import matplotlib.pyplot as plt
import numpy
from train_model import convert_file, from_doccano

@plac.annotations(input_file=("Input file", "option", "i", str),
                  model=("Trained model to load","option","m",str),
                  doccano=("Is the Input from Doccano","flag","d"))


def main(input_file=None, model=None,doccano=False):
    nlp = spacy.load(model)
    if doccano==True:
        from_doccano(input_file,'labels_test.json')
        input_file = 'labels_test.json'
    docs = convert_file(input_file,write_file=False)
    cm, _ = plot_confusion_matrix(docs, nlp, normalize=False)
    classes = get_dataset_labels(docs)
    met = metrics(cm, classes)
    print(f"Overall Accuracy: {met['accuracy']}")
    for m in met['labels']:
        print(f"{m['label']}, Precision = {m['metrics']['precision']}, Recall = {m['metrics']['recall']}")
    plt.show()


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


def create_prediction_vector(text,nlp):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text,nlp)]


def create_total_prediction_vector(docs: list,nlp):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc[0],nlp))
    return prediction_vector


def get_all_ner_predictions(text,nlp):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = biluo_tags_from_offsets(doc, entities)
    return bilou_entities


def get_dataset_labels(docs):
    labels = [item for sublist in [[r[2] for r in R[1]['entities']] for R in docs] for item in sublist]
    classes = sorted(set(labels))
    classes.append('O')
    return classes


def plot_confusion_matrix(docs, nlp, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = 'Confusion Matrix, for SpaCy NER'

    # Compute confusion matrix
    classes = get_dataset_labels(docs)
    y_true = create_total_target_vector(docs, nlp)
    y_pred = create_total_prediction_vector(docs, nlp)

    cm = confusion_matrix(y_true,y_pred,classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm, ax


def metrics(cm,classes):
    TP = [cm[i, i] for i in range(cm.shape[0])]
    FP = [sum(cm[:, i]) - cm[i, i] for i in range(cm.shape[0])]
    FN = [sum(cm[i, :]) - cm[i, i] for i in range(cm.shape[0])]
    prec = [TP[i] / (TP[i] + FP[i]) for i in range(len(TP))]
    prec = [0.0 if x != x else x for x in prec]
    rec = [TP[i] / (TP[i] + FN[i]) for i in range(len(TP))]
    acc = sum(TP) / sum([sum(cm[:,i]) for i in range(cm.shape[0])])
    lab_met = [{'label': classes[i],
                'metrics': {'precision': prec[i],
                            'recall': rec[i]}} for i in range(len(classes))]
    met = {'accuracy': acc,
           'labels':lab_met}
    with open('metrics.json', "w") as o:
        o.write(str(json.dumps(met)))
    return met


if __name__ == '__main__':
    plac.call(main)