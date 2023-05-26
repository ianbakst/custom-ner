import json
import os
import datetime

def list_all(data_dir='data'):
    """Lists all model metadata.

    """
    data = []
    sets = [i for i in os.walk(data_dir)][0][1]
    for ds in sets:
        with open(os.path.join(data_dir,ds,'meta.json'),'r') as f:
            data.append(json.load(f))
    return data


def delete_all(data_dir='data'):
    current_list = list_all(data_dir)
    for entry in current_list:
        delete(entry['name'])
    return


def delete(dname, data_dir='data'):
    os.remove(os.path.join(data_dir, dname))
    return


def initialize(data):
    try:
        name = data['name']
    except:
        return "No dataset name given.", 500
    doccano = data.get('doccano',True)
    metadata = {k: data[k] for k in data.keys() - {'data','doccano'}}
    current_list = list_all()
    names = [cd['name'] for cd in current_list]
    if name in names:
        return f"Dataset '{name}' already exists.", 501
    data = data.get('data')
    if doccano:
        new_data = convert_from_doccano(data)
    else:
        new_data = data
    return initialize_data(metadata,new_data), 204


def initialize_data(metadata,data):
    os.mkdir(os.path.join('data',metadata['name']))
    with open(os.path.join('data',metadata['name'],'data.json'),'w') as f:
        for d in data:
            f.write(str(json.dumps(d)))
            f.write('\n')
    return initialize_metadata(metadata)


def initialize_metadata(metadata, data_dir='data'):
    metadata['created'] = (datetime.datetime.now()).strftime("%d-%b-%Y %H:%M:%S")
    metadata['last_modified'] = metadata['created']
    write_metadata(metadata)
    return metadata


def write_metadata(metadata, data_dir='data'):
    with open(os.path.join(data_dir,metadata['name'],'meta.json'), 'w') as f:
        f.write(str(json.dumps(metadata)))
    return metadata


def upsert(dname, data, data_dir='data'):
    current_list = list_all()
    names = [cd['name'] for cd in current_list]
    if dname not in names:
        return f"Dataset'{dname}' does not yet exist.", 501
    new_data = convert_from_doccano(data)
    with open(os.path.join(data_dir,dname,'data.json'),'a') as f:
        for d in new_data:
            f.write('\n')
            f.write(str(json.dumps(d)))
    return update_metadata(dname)


def read_metadata(name, data_dir='data'):
    with open(os.path.join(data_dir,name,'meta.json'),'r') as f:
        metadata = json.load(f)
    return metadata


def update_metadata(name, data_dir='data'):
    metadata = read_metadata(name)
    metadata['last_modified'] = (datetime.datetime.now()).strftime("%d-%b-%Y %H:%M:%S")
    return write_metadata(metadata)


def convert_from_doccano(data):
    new_data = []
    for entry in data:
        ann = [{"label": [k[2]],
                "points": [{"text": entry['text'][k[0]:k[1]],
                            "start": k[0],
                            "end": k[1] - 1}]} for k in entry['labels']]
        text = entry['text']
        entities = []
        for a in ann:
            point = a['points'][0]
            labels = a['label']
            if not isinstance(labels, list):
                    labels = [labels]
            for label in labels:
                entities.append((point['start'], point['end'] + 1, label))
        new_data.append((text, {"entities": entities}))
    return new_data