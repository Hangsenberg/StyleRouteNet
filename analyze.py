import argparse
from pathlib import Path

import pandas as pd

from srnet.engine import summarize_results
from srnet.inference import cohort_intervals, read_predictions
from srnet.metrics import aggregate_patients, align_predictions, binary_metrics, calibration_bins, confidence_intervals
from srnet.runtime import stable_seed, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--comparator")
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--patient-aggregation", choices=["mean", "max", "noisy_or"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.bootstrap == 1 or args.comparator and args.bootstrap < 2:
        raise ValueError("Paired comparison requires at least two bootstrap replicates; otherwise use zero or at least two")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat([read_predictions(filename) for filename in args.predictions], ignore_index=True)
    if args.reference_seed is not None:
        predictions = predictions.loc[predictions["seed"] == args.reference_seed].copy()
    if predictions.empty:
        raise ValueError("No predictions remain after seed selection")
    if args.patient_aggregation:
        predictions = aggregate_patients(predictions, args.patient_aggregation)
    summarize_results(predictions, output, reference_seed=args.reference_seed)
    if args.patient_aggregation:
        pooled_patients = []
        for key, group in predictions.groupby(["dataset", "model", "seed"], sort=True):
            pooled_patients.append(dict(zip(["dataset", "model", "seed"], key),
                                        **binary_metrics(group["label"], group["score"])))
        write_csv(output / "metrics_pooled_patients.csv", pd.DataFrame(pooled_patients))
    intervals, bins = [], []
    for key, group in predictions.groupby(["dataset", "model", "seed"], sort=True):
        identifiers = dict(zip(["dataset", "model", "seed"], key))
        if args.bootstrap:
            intervals.append(dict(identifiers, **cohort_intervals(group, args.bootstrap, stable_seed(args.bootstrap_seed, key))))
        calibration = calibration_bins(group["label"], group["score"])
        for name, value in identifiers.items():
            calibration[name] = value
        bins.append(calibration)
    pd.concat(bins, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False)
    write_json(output / "pooled_oof_intervals.json", {
        "estimand": "pooled held-out AUROC conditional on the fitted models, separately for each seed",
        "patient_aggregation": args.patient_aggregation, "results": intervals,
    })
    if args.comparator:
        comparator = read_predictions(args.comparator)
        if args.reference_seed is not None:
            comparator = comparator.loc[comparator["seed"] == args.reference_seed].copy()
        if args.patient_aggregation:
            comparator = aggregate_patients(comparator, args.patient_aggregation)
        comparisons = []
        for key, group in predictions.groupby(["dataset", "model", "seed"], sort=True):
            selected = comparator.loc[(comparator["dataset"] == key[0]) & (comparator["seed"] == key[2])]
            if selected["model"].nunique() != 1:
                raise ValueError("Each comparator file must contain one model for each cohort and seed")
            aligned = align_predictions(group, selected)
            patient_ids = aligned["patient_id_first"].fillna("").astype(str)
            if patient_ids.eq("").all():
                patient_ids = None
            elif patient_ids.eq("").any():
                raise ValueError("Incomplete patient IDs for paired cluster bootstrap")
            comparison = confidence_intervals(
                aligned["label_first"], aligned["score_first"], patient_ids,
                args.bootstrap, stable_seed(args.bootstrap_seed, key, "paired"), aligned["score_second"],
            )
            comparisons.append(dict(dataset=key[0], model=key[1], seed=key[2], comparator=selected["model"].iloc[0], **comparison))
        write_json(output / "paired_comparisons.json", comparisons)
    print(output.resolve())


if __name__ == "__main__":
    main()
