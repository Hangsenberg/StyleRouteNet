import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score


def calibration_bins(labels, scores, bins=10):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > 0.5).astype(np.int64)
    confidence = np.maximum(scores, 1.0 - scores)
    correct = (predictions == labels).astype(np.float64)
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            rows.append({"bin": index, "count": int(selected.sum()), "confidence": float(confidence[selected].mean()), "accuracy": float(correct[selected].mean())})
    return pd.DataFrame(rows, columns=["bin", "count", "confidence", "accuracy"])


def binary_metrics(labels, scores, threshold=0.5, bins=10):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be equally sized nonempty vectors")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    prediction = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    result = {
        "n": int(len(labels)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
        "accuracy": divide(tp + tn, len(labels)),
        "precision": precision,
        "recall": sensitivity,
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": precision,
        "npv": divide(tn, tn + fn),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "auroc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float(np.square(scores - labels).mean()),
    }
    clipped = np.clip(scores, 1e-12, 1.0 - 1e-12)
    result["nll"] = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
    calibration = calibration_bins(labels, scores, bins)
    result["ece"] = float((calibration["count"] * (calibration["confidence"] - calibration["accuracy"]).abs()).sum() / len(labels))
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
        result["pr_auc"] = float(auc(pr_recall, pr_precision))
    return result


def select_threshold(labels, scores, rule="youden", minimum_sensitivity=0.9):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation threshold selection requires both classes")
    if rule == "fixed":
        return 0.5
    candidates = np.unique(np.r_[np.nextafter(scores.min(), -np.inf), scores, 0.5])
    rows = [binary_metrics(labels, scores, float(threshold)) for threshold in candidates]
    if rule == "youden":
        key = lambda row: (row["sensitivity"] + row["specificity"] - 1.0, -abs(row["threshold"] - 0.5), row["threshold"])
    elif rule == "f1":
        key = lambda row: (row["f1"], -abs(row["threshold"] - 0.5), row["threshold"])
    elif rule == "sensitivity":
        rows = [row for row in rows if row["sensitivity"] >= minimum_sensitivity]
        if not rows:
            raise ValueError("No validation threshold meets the sensitivity requirement")
        key = lambda row: (row["specificity"], -abs(row["threshold"] - 0.5), row["threshold"])
    else:
        raise ValueError(f"Unknown threshold selection rule: {rule}")
    return float(max(rows, key=key)["threshold"])


def confidence_intervals(labels, scores, patient_ids=None, iterations=2000, seed=2026, paired_scores=None):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    paired = None if paired_scores is None else np.asarray(paired_scores, dtype=np.float64)
    if paired is not None and paired.shape != scores.shape:
        raise ValueError("Paired predictions must have identical alignment")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUROC confidence intervals require both classes")
    if patient_ids is None:
        groups = [np.array([index]) for index in range(len(labels))]
        unit = "image"
    else:
        identifiers = np.asarray(patient_ids, dtype=str)
        if identifiers.shape != labels.shape or np.any(identifiers == ""):
            raise ValueError("Patient-cluster bootstrap requires actual patient IDs")
        groups = [np.flatnonzero(identifiers == value) for value in np.unique(identifiers)]
        unit = "patient"
    if iterations < 2 or len(groups) < 2:
        raise ValueError("Bootstrap requires at least two replicates and two independent units")
    rng = np.random.default_rng(seed)
    first, second, differences = [], [], []
    attempts = 0
    while len(first) < iterations and attempts < iterations * 50:
        attempts += 1
        selection = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selection])
        if len(np.unique(labels[indices])) != 2:
            continue
        first_value = float(roc_auc_score(labels[indices], scores[indices]))
        first.append(first_value)
        if paired is not None:
            second_value = float(roc_auc_score(labels[indices], paired[indices]))
            second.append(second_value)
            differences.append(first_value - second_value)
    if len(first) != iterations:
        raise RuntimeError("Too few valid patient-cluster bootstrap replicates")
    interval = lambda values: [float(value) for value in np.percentile(values, [2.5, 97.5])]
    result = {"unit": unit, "clusters": len(groups), "replicates": len(first), "attempts": attempts, "seed": seed,
              "auroc": float(roc_auc_score(labels, scores)), "auroc_ci95": interval(first)}
    if paired is not None:
        result.update(comparator_auroc=float(roc_auc_score(labels, paired)), comparator_auroc_ci95=interval(second),
                      auroc_difference=float(roc_auc_score(labels, scores) - roc_auc_score(labels, paired)),
                      auroc_difference_ci95=interval(differences))
    return result


def align_predictions(first, second):
    keys = ["dataset", "seed", "fold", "sample_id"]
    for frame in (first, second):
        if not set(keys + ["label", "score", "patient_id"]).issubset(frame.columns):
            raise ValueError("Prediction files are missing required columns")
        if frame.duplicated(keys).any():
            raise ValueError("Predictions contain duplicate sample keys")
    if set(map(tuple, first[keys].to_numpy())) != set(map(tuple, second[keys].to_numpy())):
        raise ValueError("Prediction files do not contain the same paired samples")
    merged = first.merge(second, on=keys, suffixes=("_first", "_second"), validate="one_to_one")
    if not np.array_equal(merged["label_first"], merged["label_second"]):
        raise ValueError("Paired labels differ")
    if not np.array_equal(merged["patient_id_first"].fillna(""), merged["patient_id_second"].fillna("")):
        raise ValueError("Paired patient IDs differ")
    return merged


def aggregate_patients(predictions, rule="mean"):
    required = {"patient_id", "patient_label", "score", "dataset", "model", "seed", "fold"}
    if not required.issubset(predictions.columns):
        raise ValueError("Patient evaluation requires independent patient_label values")
    if (predictions["patient_id"].fillna("") == "").any() or not predictions["patient_label"].isin([0, 1]).all():
        raise ValueError("Every patient requires an independent reference label and a patient ID")
    rows = []
    for key, group in predictions.groupby(["dataset", "model", "seed", "fold", "patient_id"], sort=True):
        if group["patient_label"].nunique() != 1:
            raise ValueError(f"Inconsistent patient reference for {key}")
        scores = group["score"].to_numpy(dtype=np.float64)
        if rule == "mean":
            score = scores.mean()
        elif rule == "max":
            score = scores.max()
        elif rule == "noisy_or":
            score = 1.0 - np.prod(1.0 - scores)
        else:
            raise ValueError(f"Unknown aggregation rule: {rule}")
        rows.append(dict(zip(["dataset", "model", "seed", "fold", "patient_id"], key),
                         sample_id=key[-1], label=int(group["patient_label"].iloc[0]), score=float(score), fields=len(group)))
    result = pd.DataFrame(rows)
    if result.groupby(["dataset", "model", "seed", "patient_id"])["fold"].nunique().max() != 1:
        raise ValueError("Patient images span multiple test folds")
    return result
