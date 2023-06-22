import sys
import os
import argparse
from pathlib import Path
from ann_benchmarks.datasets import get_dataset_fn, _cohere_wiki, _dbpedia_openai

from multiprocessing import freeze_support
from ann_benchmarks.main import main


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--delete-existing", action="store_true", help="If set, existing results will be deleted")
    args = parser.parse_args()

    results_dir = Path("results")

    if args.delete_existing:
        results_dir.unlink(missing_ok=True)

    results_dir.mkdir(exist_ok=True)

    # num_vectors = [1k, 2k, ..., 10k, 250k, ~500k]
    # num_vectors = list(range(1, 11, 1)) + [100, 250, 486]
    #  num_vectors = [10_000] + list(range(100_000, 1_000_000, 100_000))
    #  num_vectors = list(range(200_000, 1_000_000, 100_000))
    num_vectors = [1_000_000]

    # create the datasets if they don't exist
    for n in num_vectors:
        fn = get_dataset_fn(f"dbpedia-openai-{n//1000}k-angular")
        _dbpedia_openai(fn, n)
        for algo in ["qdrant", "pgvector"]:
            sys.argv = ["main.py", "--algorithm", algo, "--dataset", f"dbpedia-openai-{n//1000}k-angular", "--batch"]
            main() # run the benchmarking script

    # run the benchmarking script on each dataset (vary the number of vectors)
        #  for n in num_vectors:

    # run the benchmarking script on 10k dataset (vary the value of k - number of neighbors)
    # for k in [3, 5, 7, 10, 25, 51, 75, 101]:
    #     sys.argv = ["main.py", "--algorithm", "qdrant" ,"--dataset", f"cohere-10k-angular", "--batch", "--count", str(k)]
    #     main()
