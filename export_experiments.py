import pandas as pd

from ann_benchmarks import results
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.utils import compute_all_metrics


def get_run_desc(properties):
    return "%(dataset)s_%(count)d_%(distance)s" % properties


def get_dataset_from_desc(desc):
    return desc.split("_")[0]


def get_count_from_desc(desc):
    return desc.split("_")[1]


def get_dataset_label(desc):
    return "{} (k = {})".format(get_dataset_from_desc(desc), get_count_from_desc(desc))


def export_all_results():
    """Read all result files and compute all metrics to save them in a single CSV file"""
    cached_true_dist = []
    old_sdn = None

    df = pd.DataFrame(
        columns=[
            "dataset",
            "config",
            "train_vector_count",
            "k-nn",
            "epsilon",
            "largeepsilon",
            "rel",
            "qps",
            "p50",
            "p95",
            "p99",
            "p999",
            "distcomps",
            "build",
            "candidates",
            "indexsize",
            "queriessize",
        ]
    )

    for mode in ["non-batch", "batch"]:
        for properties, f in results.load_all_results(batch_mode=(mode == "batch")):
            sdn = get_run_desc(properties)
            if sdn != old_sdn:
                dataset, _ = get_dataset(properties["dataset"])
                cached_true_dist = list(dataset["distances"])
                old_sdn = sdn
            algo_ds = get_dataset_label(sdn)
            desc_suffix = "-batch" if mode == "batch" else ""
            sdn += desc_suffix
            print("Running for dataset %s" % dataset.filename)
            ms = compute_all_metrics(cached_true_dist, f, properties, recompute=True)
            row = ms[2].copy()
            row.update({"dataset": algo_ds, "config": ms[1], "train_vector_count": dataset["train"].shape[0]})
            new_df = pd.DataFrame(row, index=[0])

            df = pd.concat([df, new_df], ignore_index=True)

    # sort df by "train_vector_count" or "candidates" column:
    df.sort_values("train_vector_count").to_csv("results.csv", index=False)


if __name__ == "__main__":
    export_all_results()
