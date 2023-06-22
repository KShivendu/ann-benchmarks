import openai
import os
import pandas as pd
import numpy as np
# import pyarrow as pa
# import pyarrow.parquet as pq

from datasets import load_dataset, Dataset
from tqdm import tqdm

openai.api_key = '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'

LIMIT = 1000

tqdm.pandas()

ds = load_dataset('BeIR/dbpedia-entity', 'corpus')

LIMIT = 1_000_000 # 1 Million

ds = ds['corpus'].select(range(LIMIT))
df = ds.to_pandas()


def save_embeddings(embeddings, batch_name):
    try:
      # embeddings_np = np.array(embeddings)
      chunk_size = 100_000
      os.makedirs(f'./data/embeddings_chunk_{batch_name}', exist_ok=True)

      for i in range(0, len(embeddings), chunk_size):
          np.savez(f'./data/embeddings_chunk_{batch_name}/{i}.npz', embeddings=embeddings[i:i+chunk_size])

          # print file size:
          file_size = os.path.getsize(f'./data/embeddings_chunk_{batch_name}/{i}.npz')
          print(f"Saved {i} embeddings to disk. File size: {file_size / 1024 / 1024} MB")

      # with h5py.File('embeddings.h5', 'w') as h5file:
      #   h5file.create_dataset(f'/content/drive/MyDrive/vectordb/hdf5', data=embeddings)

    #   table = pa.Table.from_pandas(pd.DataFrame(embeddings))
    #   pq.write_table(table, f'/content/drive/MyDrive/vectordb/embeddings.parquet')

    except Exception as e:
      print(f"Couldn't save batch {batch_name} to drive {e}")

from typing import List
import time
import numpy as np

from tqdm import tqdm
# import h5py


MAX_RETRIES = 3
RETRY_DELAY = 1
SAVE_EMBEDDINGS_INTERVAL = 2500
SAVE_LATEST_EMBEDDINGS_INTERVAL = 50

embeddings = np.empty((0, 1536))
if os.path.exists('./data/embeddings_chunk_latest'):
    # embeddings = np.load('./data/embeddings_latest.npz')['embeddings']
    # load chunks
    for file in os.listdir('./data/embeddings_chunk_latest'):
        if file.endswith(".npz"):
            chunk = np.load(f'./data/embeddings_chunk_latest/{file}')['embeddings']
            embeddings = np.append(embeddings, chunk, axis=0)
    print("Loaded latest embeddings from disk")

print("Initail embeddings shape:", embeddings.shape)

def generate_openai_embeddings_batch(batch_titles, batch_texts) -> List[List[float]]:
    retries = 0
    while retries < MAX_RETRIES:
        try:
            time.sleep(0.01)
            prompts = [f"{title}\n{text}\n" for title, text in zip(batch_titles, batch_texts)]
            results = openai.Embedding.create(input=prompts, model="text-embedding-ada-002")
            embeddings = [r['embedding'] for r in results['data']]
            return embeddings
        except Exception as e:
            retries += 1
            print(f"Error {e} for batch - {batch_titles}")
            time.sleep(RETRY_DELAY)

    print(f"Retires failed for last batch")
    return [[]] * len(batch_titles)

BATCH_SIZE = 100

def generate_openai_embeddings(title_list, text_list) -> List[List[float]]:
    global embeddings
    num_batches = len(title_list) // BATCH_SIZE + 1
    for i in tqdm(range(num_batches)):
        if i < (len(embeddings) / BATCH_SIZE):
          continue # skip since it's already done!

        batch_titles = title_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_texts = text_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_embeddings = generate_openai_embeddings_batch(batch_titles, batch_texts)
        # embeddings.extend(batch_embeddings)
        embeddings = np.append(embeddings, np.array(batch_embeddings), axis=0)

        if (i + 1) % SAVE_LATEST_EMBEDDINGS_INTERVAL == 0:
            save_embeddings(embeddings, 'latest')

        if (i + 1) % SAVE_EMBEDDINGS_INTERVAL == 0:
            save_embeddings(embeddings, i + 1)

        # print(f"Covered {len(embeddings)/1000}k / 1M embeddings")
    return embeddings

print("Generating embeddings for the dataset...")
title_list = df['title'].tolist()
text_list = df['text'].tolist()
df['openai'] = generate_openai_embeddings(title_list, text_list)

# save_embeddings(embeddings, 'final')
# os.remove('./data/embeddings_latest.npz')

# Create a new dataset from the pandas dataframe:
result_ds = Dataset.from_pandas(df)
result_ds.save_to_disk(f'dbpedia-{LIMIT//1000}k-embeddings')

# publish the dataset
should_publish = input("Do you want to publish to huggingface? (y/n)")
if should_publish.lower() == "y":
    result_ds.push_to_hub('KShivendu/dbpedia-entities-openai-1M')
