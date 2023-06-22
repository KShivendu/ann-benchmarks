from datasets import load_dataset, Dataset

#  ds = load_dataset("./data/dbpedia-entities-openai-1M/")
ds = Dataset.load_from_disk("./data/dbpedia-entities-openai-1M/")

def publish():
  ds.push_to_hub('KShivendu/dbpedia-entities-openai-1M')

publish()


