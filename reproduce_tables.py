"""Recompute the supplied OOF tables; this command never trains a model."""
import argparse
import json
from pathlib import Path

import pandas as pd

from srnet.engine import summarize_results
from srnet.inference import read_predictions
from srnet.runtime import file_hash, write_json


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(root / "retained_results/five_seed_oof.csv"))
    parser.add_argument("--expected", default=str(root / "retained_results/supplement_expected.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = read_predictions(args.predictions)
    output = Path(args.output)
    summarize_results(frame, output, expected_folds=5, reference_seed=73421)
    across = pd.read_csv(output / "metrics_across_seeds.csv").set_index(["dataset", "model"])
    seeds = pd.read_csv(output / "metrics_by_seed.csv").set_index(["dataset", "model", "seed"])
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    results = []
    for table, rows, lookup, offset, suffix in (("S1", expected["summary_rows"], across, 2, "seed"),
                                               ("S2", expected["seed_rows"], seeds, 3, "fold")):
        for row in rows:
            key = (row[0].lower(), row[1]) + ((int(row[2]),) if offset == 3 else ())
            result = lookup.loc[key]
            for metric, target in zip(("accuracy", "precision", "recall", "f1", "auroc"), row[offset:]):
                value = f"{100 * result[metric + '_' + suffix + '_mean']:.2f} ± {100 * result[metric + '_' + suffix + '_sd']:.2f}"
                results.append(dict(table=table, key=list(key), metric=metric, expected=target, actual=value, match=value == target))
    matched = len(results) == 180 and all(row["match"] for row in results)
    write_json(output / "table_verification.json", dict(source_sha256=file_hash(args.predictions),
                                                        training_performed=False, cells=len(results),
                                                        all_match=matched, results=results))
    if not matched:
        raise ValueError("Recomputed cells differ from the retained supplementary tables; inspect table_verification.json")
    print(f"Verified all {len(results)} supplementary table cells: {output}")


if __name__ == "__main__":
    main()
