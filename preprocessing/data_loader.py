import zipfile

import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi


def download_data_from_kaggle():
    """Downloads the default of credit card clients dataset from Kaggle and returns a pandas dataframe
    containing the data."""
    api = KaggleApi()
    api.authenticate()

    # Download dataset:
    api.dataset_download_files("uciml/default-of-credit-card-clients-dataset", path="../data/")

    # Extract contents from zip previously downloaded:
    with zipfile.ZipFile("../data/default-of-credit-card-clients-dataset.zip", "r") as z:
        z.extractall("../data/")

    df = pd.read_csv("../data/UCI_Credit_Card.csv")
    return df
