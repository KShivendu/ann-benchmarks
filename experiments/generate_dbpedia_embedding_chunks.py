import os
import sys
import openai
import time
import numpy as np

from typing import List
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path

CHUNK_SIZE = 100_000 # Will take 10 steps to reach 1M

BATCH_SIZE = 1000
SAVE_CHUNK_EMBEDDINGS_INTERVAL = 10 # save every 10 batches

MAX_RETRIES = 3
RETRY_DELAY = 1

def load_embeddings(chunk_id):
    embeddings = np.empty((0, 1536))
    if os.path.exists(f'./data/embeddings_chunk_latest/{chunk_id}.npz'):
        embeddings = np.load(f'./data/embeddings_chunk_latest/{chunk_id}.npz')['embeddings']

    return embeddings

def save_embeddings(embeddings, chunk_id):
    try:
      embeddings_dir = Path(f'./data/embeddings_chunk_latest')
      embeddings_dir.mkdir(exist_ok=True)

      chunk_path = embeddings_dir / f'{chunk_id}.npz'

      np.savez(chunk_path, embeddings=embeddings)
      file_size = os.path.getsize(chunk_path)
      print(f"Saved {len(embeddings)} embeddings to chunk {chunk_id}. File size: {file_size / 1024 / 1024} MB")

    except Exception as e:
      print(f"Couldn't save chunk {chunk_id} to drive. Error: {e}")


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


def generate_openai_embeddings(title_list, text_list, chunk_id) -> List[List[float]]:
    global embeddings
    num_batches = len(title_list) // BATCH_SIZE + 1
    for i in tqdm(range(num_batches)):
        if i < (len(embeddings) / BATCH_SIZE):
          continue # skip since it's already done!

        batch_titles = title_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_texts = text_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        if len(batch_titles) > 0 and len(batch_texts) > 0:
            batch_embeddings = generate_openai_embeddings_batch(batch_titles, batch_texts)
            embeddings = np.append(embeddings, np.array(batch_embeddings), axis=0)

        if (i + 1) % SAVE_CHUNK_EMBEDDINGS_INTERVAL == 0:
            save_embeddings(embeddings, chunk_id)

    return embeddings

print("Generating embeddings for the dataset...")
ds = load_dataset('BeIR/dbpedia-entity', 'corpus')

try:
    chunk_id = int(sys.argv[1])
except:
    chunk_id = 0

embeddings = load_embeddings(chunk_id)
print("Initial embeddings shape:", embeddings.shape)

# offset = len(embeddings)
# load offset from args:

offset = chunk_id * CHUNK_SIZE
print(f"Starting from {offset//1000}k embedding till {(offset+ CHUNK_SIZE)//1000}k")
# select based on embeddings
ds = ds['corpus'].select(range(offset, offset + CHUNK_SIZE))

generate_openai_embeddings(ds['title'], ds['text'], chunk_id)
