import os
import logging


ee_dir = f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}"


def ask_user(question):
    check = str(input(f"{question}? (Y/N): ")).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Invalid Input.')
            return ask_user()

    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user()

def setup_logging():

    logging.basicConfig(format='%(levelname)s [%(asctime)s] %(message)s', level=logging.INFO)
    logging.captureWarnings(True)
    console_handler = logging.StreamHandler()
    logging.getLogger('py.warnings').addHandler(console_handler)
    logging.basicConfig(format='%(levelname)s [%(asctime)s] %(message)s', level=logging.DEBUG)
    return