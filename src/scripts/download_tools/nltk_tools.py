import nltk
import os

from config import datasets_dir

def setup_nltk_resources(data_dir):
    nltk.data.path.append(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    nltk.download('wordnet', download_dir=data_dir)
    print("NLTK resources are set up and ready to use")

if __name__ == '__main__':
    setup_nltk_resources(datasets_dir)

