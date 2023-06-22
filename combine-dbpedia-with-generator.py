from datasets import Dataset, load_dataset
import numpy as np
import time

LIMIT = 1_000_000
CHUNK_SIZE = 100_000 # 100k
# NUM_CHUNKS = LIMIT // CHUNK_SIZE # 10 chunks of 100k each

print("Loading dataset")
ds = load_dataset('BeIR/dbpedia-entity', 'corpus', split='corpus', streaming=False)
#  ds = ds['corpus']


# new_dataset = Dataset.from_dict({'_id': [], "title": [], "text": [], "openai": []})

def data_generator():
    counter = 0
    latest_embedding_chunk = np.load(f'./data/embeddings_chunk_latest/0.npz')['embeddings']

    print("Starting data generator")
    for entry in ds:
        #  print(entry)
        if counter % CHUNK_SIZE == 0:
            print(f"Loaded new chunk {counter // CHUNK_SIZE} after completing {counter} vectors")
            latest_embedding_chunk = np.load(f'./data/embeddings_chunk_latest/{counter // CHUNK_SIZE}.npz')['embeddings']

        entry['openai'] = latest_embedding_chunk[counter % CHUNK_SIZE].tolist()
        counter += 1

        if counter % 10_000 == 0:
            print(f"Processed {counter} entries")

        yield entry

        if counter == LIMIT:
            break


new_ds = Dataset.from_generator(data_generator)
#  new_ds.save_to_disk('./data/dbpedia-entities-openai-1M')

new_ds.push_to_hub('KShivendu/dbpedia-entities-openai-1M-generator')

# for entry in ds:


# for i in range(NUM_CHUNKS):
#     chunk_ds = ds.select(range(i*CHUNK_SIZE, (i+1)*CHUNK_SIZE))
#     chunk_embeddings = np.load(f'./data/embeddings_chunk_latest/{i}.npz')['embeddings']

#     print(f"Read embeddings from chunk {i}")
#     print(f"Shape is {chunk_embeddings.shape}")

#     print("Converting embeddings to list")
#     chunk_ds.add_column('openai', chunk_embeddings.tolist())
#     del chunk_embeddings
#     print("Loaded embeddings to ds")

