from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import time

LIMIT = 1_000_000
CHUNK_SIZE = 100_000 # 100k
NUM_CHUNKS = LIMIT // CHUNK_SIZE # 10 chunks of 100k each

ds = load_dataset('BeIR/dbpedia-entity', 'corpus')

# df = ds.to_pandas()
# del ds

# load embedding from chunks:
#  embeddings = np.empty((0, 1536))

#  final_ds = ds['corpus']

for i in range(NUM_CHUNKS):
    chunk_ds = ds['corpus'].select(range(i*CHUNK_SIZE, (i+1)*CHUNK_SIZE)) # .select(range(10))
    #  chunk_df = chunk_ds.to_pandas()
    #  del chunk_ds
    chunk_embeddings = np.load(f'./data/embeddings_chunk_latest/{i}.npz')['embeddings'] # [:10]

    print(f"Read embeddings from chunk {i}")
    print(f"Shape is {chunk_embeddings.shape}")
    # add embeddings to dataset as a new column
    print("Converting embeddings to list")
    chunk_ds.add_column('openai', chunk_embeddings.tolist())
    del chunk_embeddings
    print("Loaded embeddings to ds")

    # create a split with chunk name and push to huggingface
    chunk_ds = DatasetDict({f'corpus_chunk_{i}': chunk_ds})
    chunk_ds.push_to_hub(f'KShivendu/dbpedia-entities-openai-1M')

    time.sleep(1)
    print("Waiting after dataset")

# add embeddings to dataset as a new column
# print("Converting embeddings to list")
# df['openai'] = embeddings.tolist()
# print("Loaded embeddings to df")
# ds.add_column('embeddings', embeddings.tolist())

# save dataset to disk
# ds.save_to_disk('./data/dbpedia-with-embeddings')

# ds = Dataset.from_pandas(df)
# print(ds)

should_publish = input("Do you want to publish to huggingface? (y/n)")
if should_publish.lower() == "y":
    ds.push_to_hub('KShivendu/dbpedia-entities-openai-1M')
