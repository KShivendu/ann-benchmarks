import glob
import os

import pandas as pd


def load_parquet(parquet_path):
    return pd.read_parquet(parquet_path)

def print_parquet_data(data):
    for _, row in data.iterrows():
        # print("Abstract:", row['abstract'])
        print("Embedding:", len(row['embeddings']))
        print("DOI:", row['doi'])
        print()

if __name__ == "__main__":
    print("Starting...")
    # directory_path = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    print("Directory path:", directory_path)
    for file_path in glob.glob(os.path.join(directory_path, 'abstracts_*.parquet')):
        print("Loading and printing parquet file:", file_path)
        data = load_parquet(file_path)
        print_parquet_data(data)
