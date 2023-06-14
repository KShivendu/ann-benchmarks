import sys
import os
import argparse
from pathlib import Path
from ann_benchmarks.datasets import get_dataset_fn, _cohere_wiki

from multiprocessing import freeze_support
from ann_benchmarks.main import main


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--delete-existing", action="store_true", help="If set, existing results will be deleted")
    args = parser.parse_args()

    if args.delete_existing: # or just set this to True
        results_dir = Path("results")
        results_dir.unlink(missing_ok=True)
        results_dir.mkdir(exist_ok=True)

    # num_vectors = [1k, 2k, ..., 10k, 250k, ~500k]
    num_vectors = list(range(1, 11, 1)) + [100, 250, 486]

    # create the datasets if they don't exist
    for n in num_vectors:
        fn = get_dataset_fn(f"cohere-{n}k-angular")
        _cohere_wiki(fn, "angular", n)

    # run the benchmarking script on each dataset (vary the number of vectors)
    for n in num_vectors:
        sys.argv = ["main.py", "--algorithm", "qdrant", "--dataset", f"cohere-{n}k-angular", "--batch"]
        main() # run the benchmarking script

    # run the benchmarking script on 10k dataset (vary the value of k - number of neighbors)
    # for k in [3, 5, 7, 10, 25, 51, 75, 101]:
    #     sys.argv = ["main.py", "--algorithm", "qdrant" ,"--dataset", f"cohere-10k-angular", "--batch", "--count", str(k)]
    #     main()
