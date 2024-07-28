import os
from typing import List, Dict, Optional
import pickle
from zipfile import ZipFile

import numpy as np
import requests

from config import datasets_dir

    
def download_and_save_glove_vectors(data_dir: str, vector_dim: int=50):
    """
    Downloads the GloVe dataset and gets it ready for the project.
    
    Args:
        data_dir: Dictory for saving the intended files
        vector_dim: Number of dimensions per word
    """
    glove_filename = f'glove.6B.{vector_dim}d.txt'
    glove_url = 'http://download.cs.stanford.edu/nlp/data/glove.6B.zip'
    pickled_filepath = os.path.join(data_dir, f'glove_vectors_{vector_dim}d.pkl')
    os.makedirs(data_dir, exist_ok=True)
    glove_dir = os.path.join(data_dir, 'glove.6B')
    glove_zip_path = os.path.join(data_dir, 'glove.6B.zip')
    if not os.path.exists(glove_dir):
        print('Downloading GloVe embeddings')
        response = requests.get(glove_url)
        with open (glove_zip_path, 'wb') as f:
            f.write(response.content)
        print('Download Complete')
        print('Unzipping GloVe Embeddings')
        with ZipFile(glove_zip_path, 'r') as zip_f:
            zip_f.extractall(glove_dir)
        print('GloVe Embeddings Unzipped')
        os.remove(glove_zip_path)
    else:
        print('GloVe embeddings already available')
    glove_file_path = os.path.join(glove_dir, glove_filename)
    export_glove_dict(glove_file_path, pickled_filepath)

    
def export_glove_dict(filename: str, pickled_filepath: str)-> Dict:
    glove_vectors = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split(' ')
            word = parts[0]
            vector_components = parts[1:]
            glove_vectors[word] = vector_components
    with open(pickled_filepath, 'wb') as pkl:
        pickle.dump(glove_vectors, pkl)
    
if  __name__ == '__main__':
    download_and_save_glove_vectors(datasets_dir, vector_dim=50)