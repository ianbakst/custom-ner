import pandas as pd
import json, yaml
import requests
import logging
import math
from entity_extractor.doccano.doccanoAPI import DoccanoClient as Client

def csv2text(input_dataset, output_dataset, column_name):
    ##### Saves column from csv to txt file.

    logging.info(f'Reading input dataset {input_dataset}.')
    data = pd.read_csv(input_dataset, usecols = [column_name])

    logging.info(f'Processing input dataset {input_dataset}.')
    prod_desc = data.loc[:,column_name].unique()

    num_lines = len(prod_desc)
    logging.info(f'Processing output dataset {output_dataset} ({num_lines} unique lines).')

    with open(output_dataset, "w") as txt_file:
        for line in prod_desc:
            txt_file.write(line + "\n")

    return

def labels2dict(labels_json):
    ##### Converts a Doccano labels json to a dictionary

    labels_dict = {}
    for label in labels_json:
        labels_dict[label['id']] = label['text']

    return labels_dict

def get_annotations(labels_dict, labels):
    ##### Converts the Doccano labels dictionary to a [onset, offset, label] format

    annotations = []
    for label in labels:
        annotation = [label['start_offset'], label['end_offset'], labels_dict[label['label']]]
        annotations.append(annotation)
    return annotations

def convert_json(documents_file = None, labels_file = None):
    ##### Converts a Docanno output json to a Doccano input json

    logging.info(f'Reading documents from {documents_file}.')
    with open(documents_file) as input_file:
        document_input = json.load(input_file)

    logging.info(f'Reading labels from {labels_file}.')
    with open(labels_file) as input_file:
        labels_input = json.load(input_file)

    labels_dict = labels2dict(labels_input)

    output_file = documents_file.lower().strip('.json') + '_annotated.json'
    logging.info(f'Outputing annoted documents to {output_file}.')

    document_output = {}

    with open(output_file, 'w') as output:
        for document in document_input["results"]:
            document_output['text'] = document['text']
            document_output['labels'] = get_annotations(labels_dict, document['annotations'])
            json.dump(document_output, output)
            output.write('\n')

    return

def find_project(client, name):
    ##### Finds a Doccano project by name

    projectExists = False

    project_list = client.get_project_list()

    for project in project_list:
        if project['name'] == name and project['project_type'] == project_type:
            logging.info(f'Project {name} found!')
            return project['id']

    return False

def doccano_downloader(client, project_name, documents_file = None, labels_file = None):
    ##### Downloads a Doccano annotated project

    logging.info(f'Checking existing projects named {project_name}.')
    project_id = find_project(client, project_name)

    if project_id:
        logging.info(f'Getting project {project_name} statistics.')
        statistics = client.get_project_statistics(project_id = project_id)

        logging.info(f'Fetching documents from {project_name}.')
        documents = client.get_document_list(project_id = project_id,
                                             url_parameters = {'limit': [statistics['total']], 'offset': [0]})

        logging.info(f'Saving documents from {project_name} to {documents_file}.')
        with open(documents_file, 'w') as output_file:
            json.dump(documents, output_file)

        logging.info(f'Fetching labels from {project_name}.')
        labels = client.get_label_list(project_id=project_id)

        logging.info(f'Saving labels from {project_name} to {labels_file}.')
        with open(labels_file, 'w') as output_file:
            json.dump(labels, output_file)

    else:
        logging.info(f'Project {project_name} not found.')

    return

def doccano_uploader(client, project_name, input_dataset, batch = 1000):
    ##### Uploads data to a Doccano project

    logging.info(f'Checking existing projects named {project_name}.')
    project_id = find_project(client, project_name)

    if not project_id:
        logging.info(f'Creating {project_name}.')
        project = client.post_project(project_name = project_name,
                                      project_desc = '',
                                      project_type = config.doccano_project_type).json()
        project_id = project['id']

    logging.info(f'Uploading {input_dataset} to {project_name}.')
    if '.json' in input_dataset:
        file_format = 'json'
    else:
        file_format = 'plain'

    num_lines = sum(1 for line in open(input_dataset))
    if batch > 0:
        num_batch = math.ceil(num_lines / batch)
    else:
        num_batch = 1
        batch = num_lines
    rf = open(input_dataset, 'r')
    responses = []
    for i in range(num_batch):
        new_file = f'{input_dataset}.batch.{i}'
        wf = open(new_file, 'w')
        for i in range(batch):
            wf.write(rf.readline())
        wf.close()
        logging.info(f'Uploading batch {i} of {num_batch}.')
        responses.append(client.post_doc_upload(project_id=project_id,
                                     file_format=file_format,
                                     file_name=new_file))
        if responses[-1].status_code >= 400:
            if responses[-1].status_code == 413:
                logging.error('File-size too large. Try a smaller batch size.')
            else:
                logging.error('Error uploading. Read response.')
            return responses

    rf.close()
    return responses
