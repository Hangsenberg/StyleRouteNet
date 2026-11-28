from copy import deepcopy
from pathlib import Path

import pandas as pd

from .data import FieldDataset, dataset_fingerprint, make_loader, records_fingerprint
from .engine import finalize_predictions, predict_loader
from .metrics import aggregate_patients, binary_metrics, confidence_intervals
from .models import build_model
from .runtime import get_device, load_checkpoint, seed_everything, stable_seed


def restore_model(filename, device="auto"):
    checkpoint = load_checkpoint(filename)
    config = deepcopy(checkpoint["config"])
    seed_everything(stable_seed(checkpoint["seed"], checkpoint["fold"], "evaluation"), config["training"]["deterministic"])
    device = get_device(device)
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, config, device


def evaluation_data(config, replacement=None, source=None, manifest=None):
    data = deepcopy(config["data"])
    if replacement:
        for key in ("name", "kind", "source", "manifest", "group_by_patient", "archive_sha256", "release_commit"):
            if key in replacement:
                data[key] = replacement[key]
    if source:
        data["source"] = str(Path(source).resolve())
    if manifest:
        data["manifest"] = str(Path(manifest).resolve())
    return data


def checkpoint_partition(records, checkpoint, partition="test", data=None):
    if records_fingerprint(records) != checkpoint["records_sha256"]:
        raise ValueError("The cohort manifest differs from the checkpoint manifest")
    if checkpoint.get("data_sha256"):
        if data is None or dataset_fingerprint(records, data) != checkpoint["data_sha256"]:
            raise ValueError("The actual image content differs from the checkpoint dataset")
    if partition not in {"validation", "test"}:
        raise ValueError("Only validation and held-out test partitions can be evaluated")
    identifiers = checkpoint["split"][partition]
    mapping = {record.sample_id: record for record in records}
    if len(identifiers) != len(set(identifiers)) or not set(identifiers).issubset(mapping):
        raise ValueError("Invalid checkpoint partition identifiers")
    return [mapping[sample_id] for sample_id in identifiers]


def evaluate_records(model, checkpoint, config, data, records, device, mask_ratio=0.0, mask_shape="square", permute_styles=None, prior_strength=None, transfer=False, checkpoint_id="", partition="test"):
    dataset = FieldDataset(records, data, checkpoint["seed"], checkpoint["fold"], partition, mask_ratio, mask_shape, permute_styles)
    loader = make_loader(dataset, config["training"])
    strength = config["training"]["prior_strength"] if prior_strength is None else float(prior_strength)
    frame, metrics = predict_loader(
        model, loader, device, checkpoint["training_prior"], strength,
        data["name"], config["variant"], checkpoint["seed"], checkpoint["fold"], checkpoint_id,
    )
    threshold = float(checkpoint.get("validation_threshold", 0.5))
    frame = finalize_predictions(frame, threshold)
    if transfer:
        frame["source_dataset"] = config["data"]["name"]
        frame["source_fold"] = int(checkpoint["fold"])
        frame["fold"] = 0
    return frame, {"fixed_threshold": metrics, "source_validation_threshold": binary_metrics(frame["label"], frame["score"], threshold)}


def cohort_intervals(frame, iterations=2000, seed=2026):
    patient_ids = frame["patient_id"].fillna("").astype(str)
    if patient_ids.eq("").all():
        patient_ids = None
    elif patient_ids.eq("").any():
        raise ValueError("Patient IDs must be complete for cluster bootstrap")
    return confidence_intervals(frame["label"], frame["score"], patient_ids, iterations, seed)


def patient_results(frame, rule, threshold=0.5):
    patients = aggregate_patients(frame, rule)
    patients["prediction"] = (patients["score"] > threshold).astype(int)
    return patients, binary_metrics(patients["label"], patients["score"], threshold)


def read_predictions(filename):
    frame = pd.read_csv(filename, dtype={"sample_id": str, "patient_id": str}, keep_default_na=False)
    required = {"dataset", "model", "seed", "fold", "sample_id", "patient_id", "label", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction columns are missing: {sorted(required - set(frame.columns))}")
    missing_public_ids = frame["dataset"].eq("public") & frame["patient_id"].eq("NA")
    frame.loc[missing_public_ids, "patient_id"] = ""
    frame[["seed", "fold", "label"]] = frame[["seed", "fold", "label"]].astype(int)
    frame["score"] = frame["score"].astype(float)
    if frame.duplicated(["dataset", "model", "seed", "fold", "sample_id"]).any():
        raise ValueError("Predictions contain duplicate sample keys")
    return frame
