import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from srnet.artifacts import (collect_completed, experiment_identity, finalize_integrity, fold_signature,
                             mark_summary, partition_ids, preflight_output, read_completed_fold,
                             write_experiment_inputs)
from srnet.config import load_config
from srnet.data import FieldDataset, load_records, make_loader, records_fingerprint
from srnet.engine import finalize_predictions, summarize_results, training_prior
from srnet.metrics import binary_metrics, select_threshold
from srnet.models import style_statistics
from srnet.runtime import file_hash, write_csv, write_json
from srnet.splits import fold_records, prepare_splits


def fit_logistic(features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def objective(parameters):
        coefficients, intercept = parameters[:-1], parameters[-1]
        decision = features @ coefficients + intercept
        difference = expit(decision) - labels
        loss = np.logaddexp(0.0, decision).sum() - np.dot(labels, decision) + np.dot(coefficients, coefficients)
        gradient = np.r_[features.T @ difference + 2.0 * coefficients, difference.sum()]
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(features.shape[1] + 1, dtype=np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Logistic regression did not converge: {result.message}")
    return result


def extract_features(records, config):
    data = dict(config["data"], stain="none", permute_styles=False)
    dataset = FieldDataset(records, data, 0, 0, "test")
    loader = make_loader(dataset, config["training"])
    matrices, identifiers = [], []
    for batch in loader:
        matrices.append(style_statistics(batch["image"], config["model"]["style_eps"]).numpy().astype(np.float64))
        identifiers.extend(batch["sample_id"])
    return pd.DataFrame(np.concatenate(matrices), index=identifiers, columns=[f"{statistic}_{channel}" for statistic in ("mean", "std", "skew") for channel in ("r", "g", "b")])


def predictions_for(records, features, scaler, fitted, config, seed, fold, prior):
    matrix = scaler.transform(features.loc[[record.sample_id for record in records]].to_numpy())
    decision = matrix @ fitted.x[:-1] + fitted.x[-1]
    scores = expit(decision)
    rows = []
    for record, value, score in zip(records, decision, scores):
        rows.append({
            "dataset": config["data"]["name"], "model": "rgb_statistics_logistic", "seed": seed, "fold": fold,
            "sample_id": record.sample_id, "patient_id": record.patient_id, "patient_label": record.patient_label,
            "label": record.label, "score": float(score), "score_uncorrected": float(score),
            "prediction": int(score > 0.5), "prediction_uncorrected": int(score > 0.5),
            "logit0": 0.0, "logit1": float(value), "scaled_logit0": 0.0, "scaled_logit1": float(value),
            "adjusted_logit0": 0.0, "adjusted_logit1": float(value), "temperature": 1.0,
            "training_prior0": prior[0], "training_prior1": prior[1], "prior_strength": 0.0,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "public.yaml"))
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--split-file")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["variant"] = "rgb_statistics_logistic"
    config["model"]["pretrained_path"] = None
    config["model"]["pretrained"] = False
    if args.source:
        config["data"]["source"] = str(Path(args.source).resolve())
    if args.manifest:
        config["data"]["manifest"] = str(Path(args.manifest).resolve())
    if args.split_file:
        config["experiment"]["split_file"] = str(Path(args.split_file).resolve())
    records = load_records(config["data"])
    assignments = prepare_splits(records, config)
    seeds = config["experiment"]["seeds"] if args.seeds is None else args.seeds
    folds = list(range(1, 6)) if args.folds is None else args.folds
    if not seeds or not folds or len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError("Seeds and folds must be nonempty and unique")
    if not set(seeds).issubset(config["experiment"]["seeds"]) or not set(folds).issubset(range(1, 6)):
        raise ValueError("Requested seeds or folds are outside the shared split protocol")
    output = Path(args.output) if args.output else Path(config["experiment"]["output"]) / "rgb_statistics_logistic"
    identity = experiment_identity(records, assignments, config)
    pairs = {(int(seed), int(fold)) for seed in seeds for fold in folds}
    preflight_output(output, records, assignments, config, identity[0], pairs, args.resume, args.overwrite)
    write_experiment_inputs(output, records, assignments, config, *identity)
    features = None
    for seed in seeds:
        for fold in folds:
            folder = output / f"seed_{seed}" / f"fold_{fold}"
            split_ids = partition_ids(records, assignments, config, seed, fold)
            signature = fold_signature(identity[0], seed, fold, split_ids)
            metadata_file = folder / "run.json"
            if args.resume and not args.overwrite and metadata_file.exists():
                import json

                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                if metadata.get("status") == "complete":
                    read_completed_fold(folder, records, assignments, config, seed, fold, signature)
                    continue
            if features is None:
                features = extract_features(records, config)
                write_csv(output / "rgb_statistics.csv", features.rename_axis("sample_id"), index=True)
            training, validation, testing = fold_records(records, assignments, seed, fold)
            folder.mkdir(parents=True, exist_ok=True)
            write_json(metadata_file, {"schema": 2, "signature": signature, "status": "running", "config": config})
            scaler = StandardScaler().fit(features.loc[[record.sample_id for record in training]].to_numpy())
            train_matrix = scaler.transform(features.loc[[record.sample_id for record in training]].to_numpy())
            fitted = fit_logistic(train_matrix, [record.label for record in training])
            temporary = folder / "model.npz.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, mean=scaler.mean_, scale=scaler.scale_, coefficients=fitted.x[:-1], intercept=fitted.x[-1])
            temporary.replace(folder / "model.npz")
            prior = training_prior(training)
            validation_predictions = predictions_for(validation, features, scaler, fitted, config, seed, fold, prior)
            threshold = select_threshold(validation_predictions["label"], validation_predictions["score"], config["training"]["threshold_rule"], config["training"]["minimum_sensitivity"])
            validation_predictions = finalize_predictions(validation_predictions, threshold)
            test_predictions = finalize_predictions(predictions_for(testing, features, scaler, fitted, config, seed, fold, prior), threshold)
            checkpoint_id = file_hash(folder / "model.npz")
            for name, frame in (("validation", validation_predictions), ("test", test_predictions)):
                frame["checkpoint"] = checkpoint_id
                write_csv(folder / f"{name}_predictions.csv", frame)
            metadata = {
                "schema": 2, "signature": signature, "experiment_signature": identity[0],
                "status": "complete", "config": config, "data_sha256": identity[1],
                "seed": seed, "fold": fold, "objective": "summed binary log loss plus squared coefficient norm; unpenalized intercept",
                "converged": bool(fitted.success), "iterations": int(fitted.nit), "loss": float(fitted.fun),
                "validation_threshold": threshold, "records_sha256": records_fingerprint(records), "checkpoint_sha256": checkpoint_id,
                "split": split_ids,
                "validation_metrics": binary_metrics(validation_predictions["label"], validation_predictions["score"]),
                "test_metrics": binary_metrics(test_predictions["label"], test_predictions["score"]),
            }
            finalize_integrity(folder, metadata, "model.npz")
    predictions = collect_completed(output, records, assignments, config, identity[0])
    summarize_results(predictions, output, config["experiment"]["folds"], config["experiment"].get("reference_seed"))
    mark_summary(output, config, identity[0], predictions)
    print(output.resolve())


if __name__ == "__main__":
    main()
