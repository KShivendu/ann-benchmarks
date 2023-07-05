## PGVector vs Qdrant benchmarks:

### Setup:
```bash
git clone https://github.com/KShivendu/ann-benchmarks
cd ann-benchmarks
pip install -r requirements.txt
python install.py --algorithm qdrant # builds a container with python and qdrant
python install.py --algorithm pgvector # builds a container with python and pgvector
```

### Run:
- First limit the number of configurations that you want to run by
altering the `algorithms/{pgvector,qdrant}/config.yml` files. Otherwise, this can
take really long time to run. For example, we used the following config:
```yaml
# algorithms/pgvector/config.yml
# PGVector(lists=200, probes=2)
float:
  any:
  - base_args: ['@metric']
    constructor: PGVector
    disabled: false
    docker_tag: ann-benchmarks-pgvector
    module: ann_benchmarks.algorithms.pgvector
    name: pgvector
    run_groups:
      pgvector:
        args: [[200]]
        query_args: [[2]]
```

```yaml
# algorithms/qdrant/config.yml
# Qdrant(quantization=False, m=16, ef_construct=128, grpc=True, hnsw_ef=None, rescore=True)
float:
  any:
  - base_args: ['@metric']
    constructor: Qdrant
    disabled: false
    docker_tag: ann-benchmarks-qdrant
    module: ann_benchmarks.algorithms.qdrant
    name: qdrant
    run_groups:
      default:
        args: [
          [False], # quantization
          [ 16 ], # m
          [ 128 ], # ef_construct
        ]
        query-args: [
          [ 8 ], # hnsw_ef
          [ False ], # re-score
        ]
```

- You must have ~17GB RAM for Qdrant and ~14GB for PgVector when running against the full dataset [KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) dataset. You also need 70-80GB of disk space.
- Now run `python experiment.py`. This script runs against subsets of the dbpedia dataset covering - 100k, 200k, ..., 1M vectors in the DB. You can rerun this script at a later time and it will resume from where it left off.
- Once you ran all the experiments you were interested in, just run `python data_export.py --out res.csv`

### Results:
- Exported results in a [sheet](https://docs.google.com/spreadsheets/d/1t2-tXID2LJCXdLv1JTPQaYhmMs6woOnK7W7nkEuDsUc/edit#gid=0)
- [Tweet](https://twitter.com/NirantK/status/1674110063286571008) with the conclusions and visualizations
