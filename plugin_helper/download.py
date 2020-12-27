import requests
import sys
import os
import zipfile

def get_model(model_url, container):
    is_correct = all([os.path.isfile(item) for item in container])
    if not is_correct:
        folder_path = os.path.dirname(container[0])
        path_to_zip_file = os.path.join(folder_path, '', os.path.basename(folder_path) + '.zip')
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        download_url(model_url, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        os.remove(path_to_zip_file)

def download_url(url, save_path, chunk_size=4096):
    with open(save_path, "wb") as f:
        print("Downloading %s" % save_path)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=chunk_size):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                percent = int(0.5 + 100*dl / total_length)
                sys.stdout.write("\r[%s%s]%s%%" % ('=' * done + '>', ' ' * (50 - done), percent))
                sys.stdout.flush()