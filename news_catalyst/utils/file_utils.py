import os
import glob
from config import *
import pandas as pd


def create_output_directories():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def load_csv(directory, file_name):
    path = os.path.join(directory, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None


def store_csv(directory, file_name, df):
    if df is None:
        return
    path = os.path.join(directory, file_name)
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(directory):
        df.to_csv(path)


def delete_file(directory, file_name):
    path = os.path.join(directory, file_name)
    if os.path.exists(path):
        os.remove(path)


def delete_files_in_directory(dir_path):
    # Get all files in the directory
    files = glob.glob(os.path.join(dir_path, "*"))

    for file_path in files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def get_os_variable(key):
    if key in os.environ:
        return os.environ[key]
    else:
        print(f"{key} not set as OS environment variable - exiting")
        exit(0)