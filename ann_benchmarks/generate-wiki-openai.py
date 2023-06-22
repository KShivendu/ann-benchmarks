from typing import List
from datasets import load_dataset, Dataset
from tqdm import tqdm

import openai

LIMIT = 1000

tqdm.pandas()

ds = load_dataset('Cohere/wikipedia-22-12-simple-embeddings', split="train")
# get first LIMIT samples:
ds = ds.select(range(LIMIT)).rename_column('emb', 'cohere')
# create pandas dataframe from the dataset:
df = ds.to_pandas()

import time

def generate_openai_embeddings(title, text) -> List[float]: # 1536 dimensions
    try:
        time.sleep(0.1)
        prompt = f"{title}\n{text}\n"
        # return np.random.rand(1536).tolist()
        result = openai.Embedding.create(input=prompt, model="text-embedding-ada-002")
        return result['data'][0]['embedding']
    except Exception as e:
        print(f"Error {e} for entry - {title}: {text}")
        return []

print("Generating embeddings for the dataset...")
# generate embeddings for each example in the dataset: (progress_apply will show progress bar because of tqdm)
df['openai'] = df.progress_apply(lambda row: generate_openai_embeddings(row['title'], row['text']), axis=1)

# Create a new dataset from the pandas dataframe:
result_ds = Dataset.from_pandas(df)
result_ds.save_to_disk(f'wikipedia-{LIMIT//1000}k-embeddings')

# publish the dataset
should_publish = input("Do you want to publish to huggingface? (y/n)")
if should_publish.lower() == "y":
    result_ds.push_to_hub('KShivendu/wikipedia-1k-cohere-openai-embeddings')
