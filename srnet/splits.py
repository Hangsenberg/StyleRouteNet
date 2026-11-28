from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import records_fingerprint
from .runtime import write_json


def validate_assignments(frame, records, seeds, folds, grouped):
    required = {"sample_id", "patient_id", "label", "seed", "fold"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split assignments require {sorted(required)}")
    expected = {record.sample_id: record for record in records}
    for seed in seeds:
        selected = frame.loc[frame["seed"] == int(seed)]
        if selected["sample_id"].duplicated().any() or set(selected["sample_id"]) != set(expected):
            raise ValueError(f"Split assignments do not cover each image exactly once for seed {seed}")
        if set(selected["fold"]) != set(range(1, folds + 1)):
            raise ValueError(f"Incomplete fold identifiers for seed {seed}")
        for row in selected.itertuples(index=False):
            record = expected[row.sample_id]
            if int(row.label) != record.label or str(row.patient_id) != record.patient_id:
                raise ValueError(f"Manifest mismatch for {row.sample_id}")
        if grouped and selected.groupby("patient_id")["fold"].nunique().max() != 1:
            raise ValueError(f"Patient leakage in seed {seed}")
        for fold in range(1, folds + 1):
            validation_fold = fold % folds + 1
            for role, subset in {
                "test": selected.loc[selected["fold"] == fold],
                "validation": selected.loc[selected["fold"] == validation_fold],
                "training": selected.loc[~selected["fold"].isin([fold, validation_fold])],
            }.items():
                if set(subset["label"]) != {0, 1}:
                    raise ValueError(f"Both classes are required in {role}, seed {seed}, fold {fold}")


def prepare_splits(records, config):
    experiment = config["experiment"]
    seeds = [int(seed) for seed in experiment["seeds"]]
    folds = int(experiment["folds"])
    if folds != 5:
        raise ValueError("The experimental protocol uses five folds")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if experiment.get("reference_seed") not in seeds:
        raise ValueError("reference_seed must occur in experiment.seeds")
    mode = experiment.get("split_mode", "replay")
    if mode not in {"replay", "generate"}:
        raise ValueError("split_mode must be replay or generate")
    filename = Path(experiment["split_file"])
    grouped = config["data"].get("kind") == "clinical" or config["data"].get("group_by_patient", False)
    if filename.exists():
        frame = pd.read_csv(filename, dtype={"sample_id": str, "patient_id": str}, keep_default_na=False)
        frame[["seed", "fold", "label"]] = frame[["seed", "fold", "label"]].astype(int)
    else:
        if mode == "replay":
            raise FileNotFoundError(f"Retained split assignments are required: {filename}. Use prepare_protocol.py to import the supplied OOF assignments. New experiments must explicitly set split_mode: generate and use a separate output directory.")
        labels = np.asarray([record.label for record in records], dtype=np.int64)
        groups = np.asarray([record.patient_id for record in records])
        if grouped and (any(not value for value in groups) or len(set(groups)) < folds):
            raise ValueError("Clinical splitting requires at least five actual patient groups")
        rows = []
        for seed in seeds:
            if grouped:
                splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
                allocations = splitter.split(np.zeros(len(records)), labels, groups)
            else:
                splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                allocations = splitter.split(np.zeros(len(records)), labels)
            assigned = np.zeros(len(records), dtype=np.int64)
            for fold, (_, indices) in enumerate(allocations, 1):
                assigned[indices] = fold
            for record, fold in zip(records, assigned):
                rows.append({"sample_id": record.sample_id, "patient_id": record.patient_id, "label": record.label, "seed": seed, "fold": int(fold)})
        frame = pd.DataFrame(rows)
        validate_assignments(frame, records, seeds, folds, grouped)
        filename.parent.mkdir(parents=True, exist_ok=True)
        temporary = filename.with_suffix(filename.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(filename)
        write_json(filename.with_suffix(".json"), {
            "records_sha256": records_fingerprint(records),
            "grouped_by_patient": grouped,
            "seeds": seeds,
            "folds": folds,
            "validation_rule": "test_fold modulo folds plus one",
            "fold_numbering": "splitter_output_order",
        })
    validate_assignments(frame, records, seeds, folds, grouped)
    return frame


def fold_records(records, assignments, seed, fold, folds=5):
    selected = assignments.loc[assignments["seed"] == int(seed)]
    mapping = dict(zip(selected["sample_id"], selected["fold"]))
    validation_fold = int(fold) % folds + 1
    training, validation, testing = [], [], []
    for record in records:
        assigned = int(mapping[record.sample_id])
        if assigned == fold:
            testing.append(record)
        elif assigned == validation_fold:
            validation.append(record)
        else:
            training.append(record)
    return training, validation, testing
