import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from srnet.engine import finalize_predictions, summarize_results
from srnet.inference import read_predictions
from srnet.metrics import binary_metrics, select_threshold
from srnet.runtime import write_json


def fit_temperature(labels, margins):
    labels = np.asarray(labels, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)

    def objective(log_temperature):
        scaled = margins / np.exp(log_temperature)
        return float((np.logaddexp(0.0, scaled) - labels * scaled).mean())

    result = minimize_scalar(objective, method="bounded", bounds=(np.log(0.05), np.log(20.0)), options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError("Validation temperature fitting failed")
    return float(np.exp(result.x))


def apply_temperature(frame, temperature):
    result = frame.copy()
    result["score_before_calibration"] = result["score"]
    if "validation_threshold" in result:
        result["threshold_before_calibration"] = result["validation_threshold"]
    result["posthoc_temperature"] = temperature
    result["score"] = expit((result["adjusted_logit1"] - result["adjusted_logit0"]) / temperature)
    result["prediction"] = (result["score"] > 0.5).astype(int)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--threshold-rule", choices=["fixed", "youden", "f1", "sensitivity"], default="youden")
    parser.add_argument("--minimum-sensitivity", type=float, default=0.9)
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validation = pd.concat([read_predictions(filename) for filename in args.validation], ignore_index=True)
    testing = pd.concat([read_predictions(filename) for filename in args.test], ignore_index=True)
    keys = ["dataset", "model", "seed", "fold"]
    required = {"adjusted_logit0", "adjusted_logit1", "checkpoint"}
    for frame in (validation, testing):
        if not required.issubset(frame.columns) or frame.duplicated(keys + ["sample_id"]).any():
            raise ValueError("Calibration requires unique per-fold predictions and original adjusted logits")
        if "posthoc_temperature" in frame:
            raise ValueError("Use the original uncalibrated predictions")
    if set(map(tuple, validation[keys].to_numpy())) != set(map(tuple, testing[keys].to_numpy())):
        raise ValueError("Validation and test files must contain identical model, seed, and fold keys")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    test_frames, validation_frames, results = [], [], []
    for key, test in testing.groupby(keys, sort=True):
        selected = np.ones(len(validation), dtype=bool)
        for name, value in zip(keys, key):
            selected &= validation[name].to_numpy() == value
        val = validation.loc[selected]
        if set(test["sample_id"]) & set(val["sample_id"]):
            raise ValueError("Validation and test images overlap within a fold")
        test_patients = set(test["patient_id"]) - {""}
        val_patients = set(val["patient_id"]) - {""}
        if test_patients & val_patients:
            raise ValueError("Validation and test patients overlap within a fold")
        if test["checkpoint"].nunique() != 1 or val["checkpoint"].nunique() != 1 or test["checkpoint"].iloc[0] != val["checkpoint"].iloc[0]:
            raise ValueError("Validation and test predictions must come from the same frozen checkpoint")
        temperature = fit_temperature(val["label"], val["adjusted_logit1"] - val["adjusted_logit0"])
        calibrated_val = apply_temperature(val, temperature)
        calibrated_test = apply_temperature(test, temperature)
        threshold = select_threshold(calibrated_val["label"], calibrated_val["score"], args.threshold_rule, args.minimum_sensitivity)
        validation_frames.append(finalize_predictions(calibrated_val, threshold))
        test_frames.append(finalize_predictions(calibrated_test, threshold))
        results.append(dict(zip(keys, key), temperature=temperature, validation_threshold=threshold,
                            before=binary_metrics(test["label"], test["score"]), after=binary_metrics(calibrated_test["label"], calibrated_test["score"]),
                            validation_threshold_metrics=binary_metrics(calibrated_test["label"], calibrated_test["score"], threshold)))
    pd.concat(validation_frames, ignore_index=True).to_csv(output / "validation_predictions.csv", index=False)
    summarize_results(pd.concat(test_frames, ignore_index=True), output, reference_seed=args.reference_seed)
    write_json(output / "calibration.json", {"fitting_partition": "validation only", "temperature_bounds": [0.05, 20.0], "threshold_rule": args.threshold_rule, "results": results})
    print(output.resolve())


if __name__ == "__main__":
    main()
