"""Summarize the paired Study One training seeds and validation selection."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from train_vgg16 import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("study_one/results"))
    root = parser.parse_args().results
    config = json.loads((root / "configuration.json").read_text())
    metrics = ("accuracy", "defect_precision", "defect_recall", "defect_f1", "roc_auc")
    seed_rows = []
    summary = []
    for variant, losses in config["validation_losses"].items():
        rows = []
        for seed, loss in zip(config["seeds"], losses, strict=True):
            result = json.loads((root / variant / str(seed) / "metrics.json").read_text())
            row = dict(variant=variant, seed=seed, validation_loss=loss,
                       n_test=result["n_test"], missed_defects=result["confusion_matrix"][1][0],
                       **{metric: result[metric] for metric in metrics})
            seed_rows.append(row)
            rows.append(row)
        row = {"variant": variant, "selected": variant == config["selected_variant"]}
        for metric in ("validation_loss", *metrics):
            values = [row[metric] for row in rows]
            row[metric + "_mean"] = mean(values)
            row[metric + "_sd"] = stdev(values)
        summary.append(row)
    write_csv(root / "per_seed.csv", list(seed_rows[0]), seed_rows)
    write_csv(root / "summary.csv", list(summary[0]), summary)


if __name__ == "__main__":
    main()
