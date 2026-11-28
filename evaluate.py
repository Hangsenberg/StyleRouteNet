import argparse
from pathlib import Path

import pandas as pd

from srnet.config import load_config
from srnet.data import load_records
from srnet.inference import checkpoint_partition, cohort_intervals, evaluate_records, evaluation_data, patient_results, restore_model
from srnet.runtime import file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=[0.0])
    parser.add_argument("--mask-shape", choices=["square", "rectangle", "ellipse"], default="square")
    parser.add_argument("--permute-styles", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prior-strength", type=float)
    parser.add_argument("--patient-aggregation", choices=["mean", "max", "noisy_or"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.bootstrap < 0 or args.bootstrap == 1:
        raise ValueError("Bootstrap replicates must be zero or at least two")
    model, checkpoint, config, device = restore_model(args.checkpoint, args.device)
    replacement = load_config(args.config)["data"] if args.config else None
    data = evaluation_data(config, replacement, args.source, args.manifest)
    records = checkpoint_partition(load_records(data), checkpoint, args.partition, data)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation" / args.partition
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_id = file_hash(args.checkpoint)
    summaries = []
    for ratio in sorted(set(args.mask_ratios)):
        folder = output / f"mask_{ratio:g}_{args.mask_shape}"
        folder.mkdir(parents=True, exist_ok=True)
        predictions, metrics = evaluate_records(
            model, checkpoint, config, data, records, device,
            ratio, args.mask_shape, args.permute_styles, args.prior_strength, checkpoint_id=checkpoint_id, partition=args.partition,
        )
        predictions.to_csv(folder / "predictions.csv", index=False)
        metadata = {
            "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checkpoint_id,
            "partition": args.partition, "mask_target_ratio": ratio, "mask_shape": args.mask_shape,
            "mask_actual_ratio_mean": float(predictions["mask_actual_ratio"].mean()),
            "style_permutation": args.permute_styles if args.permute_styles is not None else data.get("permute_styles", False),
            "data": data, "metrics": metrics,
        }
        if args.bootstrap:
            metadata["auroc_interval"] = cohort_intervals(predictions, args.bootstrap, args.bootstrap_seed)
        if args.patient_aggregation:
            patients, patient_metrics = patient_results(predictions, args.patient_aggregation)
            patients.to_csv(folder / "patient_predictions.csv", index=False)
            metadata.update(patient_aggregation=args.patient_aggregation, patient_metrics=patient_metrics)
        write_json(folder / "metrics.json", metadata)
        summaries.append(dict(mask_ratio=ratio, mask_shape=args.mask_shape, **metrics["fixed_threshold"]))
    pd.DataFrame(summaries).to_csv(output / "mask_summary.csv", index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()
