import os
import shutil
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def load_kaggle_dataset(dataset, extract_folder_path, csv_name):
    # Initialize API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    extracted_folder_path = extract_folder_path
    csv_name = extract_folder_path+'/'+csv_name
    print(csv_name)
    if not os.path.exists(extract_folder_path):
        dataset = dataset
        

        api.dataset_download_files(dataset, path=extracted_folder_path, unzip=True)

    df = pd.read_csv(csv_name)
    return df

def clean_up(extracted_folder_path):
    if os.path.exists(extracted_folder_path):
        shutil.rmtree(extracted_folder_path)