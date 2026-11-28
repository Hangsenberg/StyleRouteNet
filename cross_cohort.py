import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from srnet.config import load_config
from srnet.data import load_records
from srnet.inference import cohort_intervals, evaluate_records, evaluation_data, patient_results, restore_model
from srnet.metrics import binary_metrics
from srnet.runtime import file_hash, fingerprint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--patient-aggregation", choices=["mean", "max", "noisy_or"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.bootstrap == 1:
        raise ValueError("Bootstrap replicates must be zero or at least two")
    target_config = load_config(args.target_config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    source_identity = None
    for index, filename in enumerate(args.checkpoints):
        model, checkpoint, config, device = restore_model(filename, args.device)
        identity = (config["data"]["name"], config["variant"], checkpoint["seed"])
        if source_identity is not None and identity != source_identity:
            raise ValueError("An ensemble must contain one source cohort, one model variant, and one seed")
        source_identity = identity
        data = evaluation_data(config, target_config["data"], args.source, args.manifest)
        if Path(data["source"]).resolve() == Path(config["data"]["source"]).resolve():
            raise ValueError("The transfer target must be a different source dataset")
        records = load_records(data)
        checkpoint_id = file_hash(filename)
        predictions, metrics = evaluate_records(model, checkpoint, config, data, records, device, transfer=True, checkpoint_id=checkpoint_id)
        predictions.to_csv(output / f"member_{index + 1}_predictions.csv", index=False)
        if frames:
            aligned = predictions.set_index("sample_id").loc[frames[0]["sample_id"]].reset_index()
            if not np.array_equal(aligned["label"], frames[0]["label"]) or not np.array_equal(aligned["patient_id"], frames[0]["patient_id"]):
                raise ValueError("Ensemble members use different target records")
            predictions = aligned
        frames.append(predictions)
        sources.append({
            "path": str(Path(filename).resolve()), "sha256": checkpoint_id, "seed": checkpoint["seed"],
            "fold": checkpoint["fold"], "epoch": checkpoint["epoch"] + 1,
            "training_prior": checkpoint["training_prior"], "validation_metrics": checkpoint["validation_metrics"],
            "target_metrics": metrics["fixed_threshold"],
        })
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len({source["sha256"] for source in sources}) != len(sources):
        raise ValueError("Duplicate checkpoints are not allowed in an ensemble")
    if len({source["fold"] for source in sources}) != len(sources):
        raise ValueError("Use at most one validation-selected checkpoint per source fold")
    result = frames[0][["dataset", "model", "seed", "fold", "sample_id", "patient_id", "patient_label", "label", "source_dataset"]].copy()
    result["score"] = np.stack([frame["score"].to_numpy() for frame in frames]).mean(axis=0)
    result["score_uncorrected"] = np.stack([frame["score_uncorrected"].to_numpy() for frame in frames]).mean(axis=0)
    result["prediction"] = (result["score"] > 0.5).astype(int)
    result["prediction_uncorrected"] = (result["score_uncorrected"] > 0.5).astype(int)
    result["checkpoint"] = fingerprint([source["sha256"] for source in sources])
    result["ensemble_size"] = len(frames)
    result["source_folds"] = ",".join(str(source["fold"]) for source in sources)
    result.to_csv(output / "target_predictions.csv", index=False)
    metadata = {
        "source_dataset": source_identity[0], "target_dataset": data["name"], "model": source_identity[1],
        "source_seed": source_identity[2], "members": sources, "data": data,
        "aggregation": "arithmetic mean of source-prior-corrected positive probabilities",
        "evaluation": "complete target cohort at a fixed threshold of 0.5",
        "target_fitting": False, "metrics": binary_metrics(result["label"], result["score"]),
    }
    if args.bootstrap:
        metadata["auroc_interval"] = cohort_intervals(result, args.bootstrap, args.bootstrap_seed)
    if args.patient_aggregation:
        patients, patient_metrics = patient_results(result, args.patient_aggregation)
        patients.to_csv(output / "target_patient_predictions.csv", index=False)
        metadata.update(patient_aggregation=args.patient_aggregation, patient_metrics=patient_metrics)
    write_json(output / "transfer.json", metadata)
    print(output.resolve())


if __name__ == "__main__":
    main()
