import sys
import argparse
from pathlib import Path

# change cwd to the parent of the current directory
import os
os.chdir("../")

import sys
sys.path.append(".") # important for ann_benchmarks import to work

from ann_benchmarks.datasets import get_dataset_fn, dbpedia_entities_openai_1M

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

    num_vectors = range(100_000, 1_100_000, 100_000) # 100k to 1M

    # sys.argv = ["main.py", "--algorithm", "qdrant", "--dataset", "random-xs-20-angular", "--batch"]
    # main() # run the benchmarking script

    # create the datasets if they don't exist
    for n in num_vectors:
        dataset_name = f"dbpedia-openai-{n//1000}k-angular"
        fn = get_dataset_fn(dataset_name)
        dbpedia_entities_openai_1M(fn, n)
        for algo in ["qdrant", "pgvector"]:
            sys.argv = ["main.py", "--algorithm", algo, "--dataset", dataset_name, "--batch"]
            main() # run the benchmarking script

    # run the benchmarking script on 10k dataset (vary the value of k - number of neighbors)
    # for k in [3, 5, 7, 10, 25, 51, 75, 101]:
    #     sys.argv = ["main.py", "--algorithm", "qdrant" ,"--dataset", f"cohere-10k-angular", "--batch", "--count", str(k)]
    #     main()
