import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

from config import PROJECT_DIR, datasets_dir

if __name__ == '__main__':
    
    kaggle_json_downloaded_path = os.path.expanduser('~/Downloads/kaggle.json')
    config_path = os.path.join(PROJECT_DIR, 'src', 'config')
    kaggle_json_move_path = os.path.join(config_path, 'kaggle.json')

    os.makedirs(config_path, exist_ok=True)
    # Move json with kaggle api key to project's config dir
    if not os.path.exists(kaggle_json_move_path):
        if os.path.exists(kaggle_json_downloaded_path):
            os.rename(kaggle_json_downloaded_path, kaggle_json_move_path) #Move file
            os.chmod(kaggle_json_move_path, 0o600) # Set permissions
    # Set environment variable for Kaggle api to find the json file
    os.environ['KAGGLE_CONFIG_DIR'] = config_path
    # Download FER2013 dataset
    fer2013_kgl_url = 'msambare/fer2013'
    fer_download_path = os.path.join(datasets_dir, 'FER2013')
    os.makedirs(fer_download_path, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(fer2013_kgl_url, path=fer_download_path, unzip=True)
    # Remove zip after download
    zip_path = os.path.join(datasets_dir, 'fer2013.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("FER2013 data downloaded successfully")