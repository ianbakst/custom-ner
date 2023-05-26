import json


def from_docano(infile, outfile = 'labels.json'):
    f = open(infile,'r')
    with open(outfile, "w") as o:
        for line in f:
            entry = json.loads(line)
            ann = [{"label": [k[2]],"points": [{"text":entry['text'][k[0]:k[1]],"start":k[0],"end":k[1]-1}]} for k in entry['labels']]
            new = {"content": entry['text'],
                  "annotation": ann}
            o.write(str(json.dumps(new)))
            o.write('\n')
    f.close()
    return