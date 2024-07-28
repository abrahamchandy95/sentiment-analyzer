import os
import requests
from zipfile import ZipFile

from config import datasets_dir

# Metadata
_CITATION = """\
    @article{go2009twitter,
    title={Twitter sentiment classification using distant supervision},
    author={Go, Alec and Bhayani, Richa and Huang, Lei},
    journal={CS224N project report, Stanford},
    volume={1},
    number={12},
    pages={2009},
    year={2009}        
    }
"""

_DESCRIPTION = (
    "Sentiment140 consists of Twitter messages with emoticons, "
    "which are used as noisy labels for sentiment classification. "
    "For more details, please refer to the paper."
)

_URL = "http://help.sentiment140.com/home"
_DATA_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

# Set up path for the dataset
ZIP_PATH = os.path.join(datasets_dir, 'sentiment_140.zip')
os.makedirs(datasets_dir, exist_ok=True)

# CSV filenames after extraction
sentiment140_dir = os.path.join(datasets_dir, 'SENTIMENT140')
os.makedirs(sentiment140_dir, exist_ok=True)

TRAIN_CSV = os.path.join(
    sentiment140_dir, 'training.1600000.processed.noemoticon.csv'
)
TEST_CSV = os.path.join(
    sentiment140_dir, 'testdata.manual.2009.06.14.csv'
)


def download_and_extract_twitter_data():
    if not os.path.exists(TRAIN_CSV) and not os.path.exists(TEST_CSV):
        response = requests.get(_DATA_URL)
        with open(ZIP_PATH, 'wb') as f:
            f.write(response.content)
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(datasets_dir)
        os.remove(ZIP_PATH)
        
if __name__ =='__main__':
    download_and_extract_twitter_data()
