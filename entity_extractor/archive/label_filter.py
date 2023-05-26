import json
import plac

@plac.annotations(filename=("Input File", "option", "i", str),
                  cutoff=("Minimum number of labels cutoff","option","c",int))

def main(filename=None,cutoff=10):
    clean_labels(filename,cutoff)
    return


def clean_labels(input_file,cutoff=10):
    output_file = input_file.split('.')[0]+'_cleaned'+'.json'
    with open(fname,'r') as f:
        lines = f.readlines()
    all_labels = []
    for line in lines:
        labels = [l['label'][0] for l in json.loads(line)['annotation']]
        all_labels.extend(labels)

    label_list = set(all_labels)
    unimportant = []
    for l in label_list:
        ct = all_labels.count(l)
        if ct < cutoff:
            unimportant.append(l)
    fp = open(output_file, 'w')
    for line in lines:
        L = json.loads(line)
        r = {'content': L['content']}
        annotation = []
        for a in L['annotation']:
            if a['label'][0] in unimportant:
                pass
            else:
                annotation.append(a)
        r['annotation'] = annotation
        json.dump(r, fp)
        fp.write('\n')
    return


if __name__ == '__main__':
    fname = 'example_data/best_buy_test.json'
    clean_labels(fname)