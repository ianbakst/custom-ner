# EntityExtractor

EntityExtractor is a library of scripts for Named Entity Recognition. The are split into two sub-libraries:
* **LabelMaker** contains scripts that interact with the [Doccano app](http://doccano.herokuapp.com/).
  * *csv2txt.py* saves the Product Description column from a csv to a text file.
    ```sh
      $ python scripts/LabelMaker/csv2txt.py -h
      usage: csv2txt.py [-h] [-o] input_dataset column_name
      Saves the Product Description column from a csv to a text file.
      positional arguments:
        input_dataset         Input dataset name.
        column_name           Product description column.=
        optional arguments:
        -h, --help            show this help message and exit
        -o , --output_dataset
                              Output dataset name.
      ```
  * *uploader.py* uploads a dataset from a text file to Doccano.
    ```sh
    $ python scripts/LabelMaker/uploader.py -h
    usage: uploader.py [-h] [-i]

    Loads a dataset onto Doccano.

    optional arguments:
      -h, --help            show this help message and exit
      -i , --input_dataset
                            Input dataset name.
    ```
  * *downloader.py* downloads a dataset from Doccano to a JSON.
    ```sh
    $ python scripts/LabelMaker/downloader.py -h
    usage: downloader.py [-h] [-p] [-d] [-l]

    Downloads data from a Doccano project.

    optional arguments:
      -h, --help            show this help message and exit
      -p , --project_name   Project name.
      -d , --documents_file
                            Documents output file name.
      -l , --labels_file    Labels output file name.
    ```
  * *json2json.py* converts an annotated Doccano output JSON to a Doccano input JSON.
    ```sh
    $ python scripts/LabelMaker/json2json.py -h
    usage: json2json.py [-h] [-d] [-l] [-o]

    Converts a Doccano output JSON to an input JSON.

    optional arguments:
      -h, --help            show this help message and exit
      -d , --documents_file
                            Documents file name.
      -l , --labels_file    Labels file name.
      -o , --output_file    Output file name.
    ```
  * *doccanoAPI.py* contains the REST API for the Doccano app.
