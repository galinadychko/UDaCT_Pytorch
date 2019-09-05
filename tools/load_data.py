import requests
import os
import sys
from argparse import ArgumentParser


DEFAULT_DIR = os.getcwd() + "/data/"
DEFAULT_NAME = "data.csv"
FILE_ID = '12doxMY2skT5r3yOwRV2JnDG6sv7uulwO'
URL = "https://docs.google.com/uc?export=download"
CHUNK_SIZE = 32768


def download_file_from_google_drive(id, destination):
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--save-path", "-sp", action="store", default=DEFAULT_DIR,
                        help="Path for saving file with data")
    parser.add_argument("--save-name", "-sn", action="store", default=DEFAULT_NAME,
                        help="Name for saving file with data")
    args = parser.parse_args(argv)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    download_file_from_google_drive(FILE_ID, args.save_path + args.save_name)


if __name__ == "__main__":
    main(sys.argv[1:])

