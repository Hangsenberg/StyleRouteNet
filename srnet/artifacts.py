"""Experiment identity, completed-run validation, and incremental collection."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import dataset_fingerprint, records_fingerprint
from .metrics import binary_metrics
from .runtime import file_hash, fingerprint, write_csv, write_json
from .splits import fold_records


def code_fingerprint():
    root = Path(__file__).resolve().parent.parent
    paths = sorted(list((root / "srnet").glob("*.py")) + list(root.glob("*.py")))
    return fingerprint({path.relative_to(root).as_posix(): file_hash(path) for path in paths})


def experiment_identity(records, assignments, config):
    content = dataset_fingerprint(records, config["data"])
    pretrained = config["model"].get("pretrained_path")
    signature = fingerprint({
        "schema": 2, "model": config["model"], "data": config["data"],
        "training": config["training"], "evaluation": config.get("evaluation", {}),
        "variant": config["variant"], "data_sha256": content,
        "pretrained_sha256": file_hash(pretrained) if pretrained else None,
        "protocol": {key: config["experiment"].get(key) for key in
                     ("name", "seeds", "reference_seed", "folds", "analysis", "split_mode")},
        "assignments": assignments.sort_values(["seed", "sample_id"]).to_dict("records"),
        "code": code_fingerprint(),
    })
    return signature, content


def partition_ids(records, assignments, config, seed, fold):
    partitions = fold_records(records, assignments, seed, fold, config["experiment"]["folds"])
    return dict(zip(("train", "validation", "test"), ([r.sample_id for r in p] for p in partitions)))


def fold_signature(experiment_signature, seed, fold, split):
    return fingerprint({"experiment": experiment_signature, "seed": int(seed), "fold": int(fold), "split": split})


def _read_json(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def validate_prediction_frame(frame, records, config, seed, fold, checkpoint_id):
    required = {"dataset", "model", "seed", "fold", "sample_id", "patient_id", "label", "score", "checkpoint"}
    if not required.issubset(frame.columns):
        raise ValueError("Cached predictions are missing identity, label, score, or checkpoint columns")
    expected = {r.sample_id: r for r in records}
    if frame["sample_id"].duplicated().any() or set(frame["sample_id"]) != set(expected):
        raise ValueError("Cached predictions do not cover each held-out sample exactly once")
    for key, value in {"dataset": config["data"]["name"], "model": config["variant"],
                       "seed": int(seed), "fold": int(fold), "checkpoint": checkpoint_id}.items():
        if not frame[key].eq(value).all():
            raise ValueError(f"Cached prediction {key} differs from the requested run")
    for row in frame.itertuples(index=False):
        record = expected[row.sample_id]
        if row.label != record.label or row.patient_id != record.patient_id:
            raise ValueError(f"Cached label or patient ID differs for {row.sample_id}")
        if "patient_label" in frame and row.patient_label != record.patient_label:
            raise ValueError(f"Cached independent patient label differs for {row.sample_id}")
    metrics = binary_metrics(frame["label"], frame["score"])
    if "prediction" in frame and not np.array_equal(frame["prediction"], (frame["score"] > 0.5).astype(int)):
        raise ValueError("Cached hard labels differ from the paper's fixed 0.5 rule")
    return metrics


def read_completed_fold(directory, records, assignments, config, seed, fold, signature):
    directory = Path(directory)
    metadata = _read_json(directory / "run.json")
    split = partition_ids(records, assignments, config, seed, fold)
    if metadata.get("signature") != signature or metadata.get("split") != split:
        raise ValueError(f"Saved run has different data, splits, code, or configuration: {directory}")
    if metadata.get("status") != "complete":
        raise ValueError(f"Run is unfinished: {directory}")
    checkpoint_file = metadata.get("checkpoint_file", "best.pt")
    hashes = metadata.get("files_sha256", {})
    required = {checkpoint_file, "validation_predictions.csv", "test_predictions.csv"}
    if not required.issubset(hashes):
        raise ValueError("Saved run lacks integrity hashes; use a new output directory or explicitly retrain it")
    for name, expected_hash in hashes.items():
        if Path(name).name != name or not (directory / name).is_file() or file_hash(directory / name) != expected_hash:
            raise ValueError(f"Saved artifact is missing or changed: {directory / name}")
    checkpoint_id = hashes[checkpoint_file]
    if metadata.get("checkpoint_sha256") != checkpoint_id:
        raise ValueError("Saved checkpoint digest is inconsistent")
    mapping = {r.sample_id: r for r in records}
    frames = {}
    for role in ("validation", "test"):
        frame = pd.read_csv(directory / f"{role}_predictions.csv", dtype={"sample_id": str, "patient_id": str},
                            keep_default_na=False, float_precision="round_trip")
        actual = validate_prediction_frame(frame, [mapping[sid] for sid in split[role]], config, seed, fold, checkpoint_id)
        saved = metadata[f"{role}_metrics"]
        for key, value in actual.items():
            expected = saved.get(key)
            if expected is None and np.isnan(value):
                continue
            if expected is None or not np.isclose(value, expected, rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError(f"Cached {role} metric {key} differs from its predictions")
        frames[role] = frame
    return frames["test"], metadata["test_metrics"]


def finalize_integrity(directory, metadata, checkpoint_file="best.pt"):
    directory = Path(directory)
    names = [checkpoint_file, "validation_predictions.csv", "test_predictions.csv"]
    names.extend(name for name in ("validation_patient_predictions.csv", "test_patient_predictions.csv") if (directory / name).is_file())
    metadata["checkpoint_file"] = checkpoint_file
    metadata["files_sha256"] = {name: file_hash(directory / name) for name in names}
    metadata["checkpoint_sha256"] = metadata["files_sha256"][checkpoint_file]
    write_json(directory / "run.json", metadata)


def preflight_output(output, records, assignments, config, signature, selected_pairs, resume=False, overwrite=False):
    """Check *all* saved folds before changing any top-level experiment files."""
    output = Path(output)
    registry = output / "experiment.json"
    if output.exists() and any(output.iterdir()):
        if not registry.exists() or _read_json(registry).get("signature") != signature:
            raise ValueError("Output contains a different or legacy experiment. Choose a new --output; --overwrite only replaces selected folds of the same experiment.")
    for seed in config["experiment"]["seeds"]:
        for fold in range(1, config["experiment"]["folds"] + 1):
            directory = output / f"seed_{seed}" / f"fold_{fold}"
            metadata_file = directory / "run.json"
            selected = (seed, fold) in selected_pairs
            if not metadata_file.exists():
                if directory.exists() and any(directory.iterdir()) and not (selected and overwrite):
                    raise ValueError(f"Unidentified partial run requires explicit --overwrite: {directory}")
                continue
            if selected and not resume and not overwrite:
                raise FileExistsError(f"An experiment already exists: {directory}; use --resume")
            split = partition_ids(records, assignments, config, seed, fold)
            expected = fold_signature(signature, seed, fold, split)
            metadata = _read_json(metadata_file)
            if metadata.get("signature") != expected:
                raise ValueError(f"Saved fold belongs to a different experiment: {directory}")
            if metadata.get("status") == "complete" and not (selected and overwrite):
                read_completed_fold(directory, records, assignments, config, seed, fold, expected)


def write_experiment_inputs(output, records, assignments, config, signature, content):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "input_manifest.csv", pd.DataFrame([asdict(r) for r in records]))
    write_csv(output / "split_assignments.csv", assignments)
    write_json(output / "configuration.json", config)
    write_json(output / "experiment.json", {"schema": 2, "signature": signature, "data_sha256": content,
                                           "records_sha256": records_fingerprint(records), "status": "running"})


def collect_completed(output, records, assignments, config, signature):
    frames = []
    for seed in config["experiment"]["seeds"]:
        for fold in range(1, config["experiment"]["folds"] + 1):
            directory = Path(output) / f"seed_{seed}" / f"fold_{fold}"
            metadata_file = directory / "run.json"
            if not metadata_file.exists() or _read_json(metadata_file).get("status") != "complete":
                continue
            split = partition_ids(records, assignments, config, seed, fold)
            expected = fold_signature(signature, seed, fold, split)
            frame, _ = read_completed_fold(directory, records, assignments, config, seed, fold, expected)
            frames.append(frame)
    if not frames:
        raise ValueError("No completed folds are available to summarize")
    return pd.concat(frames, ignore_index=True).sort_values(["dataset", "model", "seed", "fold", "sample_id"]).reset_index(drop=True)


def mark_summary(output, config, signature, predictions):
    completed = predictions[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"])
    expected = len(config["experiment"]["seeds"]) * config["experiment"]["folds"]
    metadata = _read_json(Path(output) / "experiment.json")
    metadata.update(signature=signature, status="complete" if len(completed) == expected else "partial",
                    completed_folds=completed.to_dict("records"), expected_folds=expected)
    write_json(Path(output) / "experiment.json", metadata)
