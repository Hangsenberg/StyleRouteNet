import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .artifacts import (code_fingerprint, collect_completed, experiment_identity, finalize_integrity,
                        fold_signature, mark_summary, partition_ids, preflight_output,
                        read_completed_fold, write_experiment_inputs)
from .data import FieldDataset, make_loader, records_fingerprint
from .losses import TwoViewObjective, corrected_logits
from .metrics import aggregate_patients, binary_metrics, select_threshold
from .models import build_model
from .runtime import append_csv, environment, file_hash, get_device, load_checkpoint, random_state, restore_random_state, save_checkpoint, seed_everything, stable_seed, write_csv, write_json
from .splits import fold_records, prepare_splits


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "specificity", "npv", "balanced_accuracy", "auroc", "pr_auc", "ece", "brier", "nll")


def model_size(model):
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    routing = getattr(model, "routing", None)
    if routing is not None:
        result["routing_parameters"] = sum(parameter.numel() for parameter in routing.parameters())
        if hasattr(routing, "target_parameters"):
            result["routing_target_parameters"] = routing.target_parameters
            result["routing_parameter_relative_error"] = abs(result["routing_parameters"] - routing.target_parameters) / routing.target_parameters
            result["attention_mlp_hidden"] = routing.hidden_dimension
    return result


def training_prior(records):
    counts = np.bincount([record.label for record in records], minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Both labels must occur in the three training folds")
    return (counts / counts.sum()).tolist()


def move_image(batch, name, device):
    image = batch.get(name)
    return None if image is None else image.to(device, non_blocking=True)


def train_epoch(model, loader, optimizer, criterion, device, scaler, amp, gradient_clip=None):
    model.train()
    totals = {name: 0.0 for name in ("loss", "classification", "consistency", "temperature")}
    count = 0
    for batch in tqdm(loader, leave=False, desc="train"):
        first = move_image(batch, "image", device)
        second = move_image(batch, "image_second", device)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            first_output = model(first, style_images=move_image(batch, "style_image", device))
            second_output = model(second, style_images=move_image(batch, "style_image_second", device))
            losses = criterion(first_output, second_output, labels)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("A nonfinite training loss was encountered")
        scaler.scale(losses["loss"]).backward()
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip), error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.shape[0]
        for name in totals:
            totals[name] += float(losses[name].detach()) * batch_size
        count += batch_size
    return {name: value / count for name, value in totals.items()}


@torch.inference_mode()
def predict_loader(model, loader, device, prior, prior_strength, dataset_name, variant, seed, fold, checkpoint=""):
    model.eval()
    rows = []
    loss_total = 0.0
    count = 0
    for batch in tqdm(loader, leave=False, desc="evaluate"):
        images = move_image(batch, "image", device)
        output = model(images, style_images=move_image(batch, "style_image", device))
        adjusted = corrected_logits(output, prior, prior_strength)
        scores = adjusted.softmax(dim=-1)[:, 1].cpu().numpy()
        original = output["scaled_logits"].softmax(dim=-1)[:, 1].cpu().numpy()
        raw_logits = output["logits"].float().cpu().numpy()
        scaled_logits = output["scaled_logits"].float().cpu().numpy()
        adjusted_logits = adjusted.cpu().numpy()
        temperatures = output["temperature"].flatten().float().cpu().numpy()
        labels = batch["label"].numpy()
        loss_total += float(F.cross_entropy(adjusted, batch["label"].to(device), reduction="sum"))
        count += len(labels)
        for index, sample_id in enumerate(batch["sample_id"]):
            rows.append({
                "dataset": dataset_name,
                "model": variant,
                "seed": int(seed),
                "fold": int(fold),
                "sample_id": sample_id,
                "patient_id": batch["patient_id"][index],
                "patient_label": int(batch["patient_label"][index]),
                "label": int(labels[index]),
                "score": float(scores[index]),
                "score_uncorrected": float(original[index]),
                "prediction": int(scores[index] > 0.5),
                "prediction_uncorrected": int(original[index] > 0.5),
                "logit0": float(raw_logits[index, 0]),
                "logit1": float(raw_logits[index, 1]),
                "scaled_logit0": float(scaled_logits[index, 0]),
                "scaled_logit1": float(scaled_logits[index, 1]),
                "adjusted_logit0": float(adjusted_logits[index, 0]),
                "adjusted_logit1": float(adjusted_logits[index, 1]),
                "temperature": float(temperatures[index]),
                "training_prior0": float(prior[0]),
                "training_prior1": float(prior[1]),
                "prior_strength": float(prior_strength),
                "mask_target_ratio": loader.dataset.mask_ratio,
                "mask_actual_ratio": float(batch["mask_fraction"][index]),
                "mask_shape": loader.dataset.mask_shape,
                "style_sample_id": batch["style_sample_id"][index],
                "checkpoint": str(checkpoint),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The evaluation partition is empty")
    metrics = binary_metrics(frame["label"], frame["score"])
    metrics["loss"] = loss_total / count
    metrics["prior_changed_predictions"] = int((frame["prediction"] != frame["prediction_uncorrected"]).sum())
    return frame, metrics


def selection_key(metrics, config, epoch):
    metric = config.get("selection_metric", "auroc")
    if metric not in metrics or not np.isfinite(metrics[metric]):
        raise ValueError(f"Invalid validation selection metric: {metric}")
    mode = config.get("selection_mode", "max")
    if mode not in {"min", "max"}:
        raise ValueError("selection_mode must be min or max")
    value = metrics[metric] if mode == "max" else -metrics[metric]
    return float(value), -float(metrics["nll"]), -int(epoch)


def finalize_predictions(frame, threshold):
    frame = frame.copy()
    frame["validation_threshold"] = float(threshold)
    frame["prediction_validation_threshold"] = (frame["score"] > threshold).astype(int)
    return frame


def fit_fold(records, assignments, config, seed, fold, output_directory, resume=False, overwrite=False, identity=None):
    output_directory = Path(output_directory)
    train_records, validation_records, test_records = fold_records(records, assignments, seed, fold, config["experiment"]["folds"])
    split_ids = partition_ids(records, assignments, config, seed, fold)
    run_signature, data_sha256 = identity or experiment_identity(records, assignments, config)
    signature = fold_signature(run_signature, seed, fold, split_ids)
    metadata_file = output_directory / "run.json"
    best_file = output_directory / "best.pt"
    last_file = output_directory / "last.pt"
    history_file = output_directory / "history.csv"
    if metadata_file.exists() and not overwrite:
        import json

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"An experiment already exists: {output_directory}")
        if metadata["signature"] != signature:
            raise ValueError("The saved run has different data, splits, code, or configuration")
        if metadata["status"] == "complete":
            return read_completed_fold(output_directory, records, assignments, config, seed, fold, signature)
        if not last_file.exists():
            raise FileNotFoundError(f"An unfinished run has no last checkpoint: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    seed_everything(stable_seed(seed, fold, "initialization"), config["training"]["deterministic"])
    device = get_device(config["training"]["device"])
    model = build_model(config, pretrained=False if resume and last_file.exists() and not overwrite else None).to(device)
    prior = training_prior(train_records)
    comparison_model = config["variant"] == "backbone" or config["variant"].startswith(("timm:", "factory:"))
    consistency_weight = 0.0 if comparison_model else config["training"]["consistency_weight"]
    criterion = TwoViewObjective(
        config["training"]["label_smoothing"], consistency_weight, config["training"]["temperature_weight"],
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=config["training"]["minimum_learning_rate"],
    )
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 0
    best_key = None
    best_epoch = -1
    if resume and last_file.exists() and not overwrite:
        saved = load_checkpoint(last_file)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint configuration does not match the requested run")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        scaler.load_state_dict(saved["scaler_state"])
        restore_random_state(saved["random_state"])
        start_epoch = saved["epoch"] + 1
        best_key = tuple(saved["best_key"]) if saved["best_key"] is not None else None
        best_epoch = saved["best_epoch"]
        if best_key is not None and not best_file.exists():
            raise FileNotFoundError("The best checkpoint is missing from an unfinished run")
        if history_file.exists() and history_file.stat().st_size:
            history = pd.read_csv(history_file)
            write_csv(history_file, history.loc[history["epoch"] <= start_epoch])
    else:
        history_file.write_text("", encoding="utf-8")
    run_metadata = {
        "schema": 2, "signature": signature, "experiment_signature": run_signature,
        "data_sha256": data_sha256, "status": "running", "config": config,
        "seed": int(seed), "fold": int(fold), "validation_fold": int(fold) % config["experiment"]["folds"] + 1,
        "training_prior": prior, "consistency_weight": consistency_weight, "split": split_ids,
        "model_size": model_size(model), "environment": environment(),
        "consistency_weight_definition": "mean maximum-class probability divided by one plus mean view temperature; temperature equals one without ETC",
        "checkpoint_selection": {"metric": config["training"]["selection_metric"], "mode": config["training"]["selection_mode"], "ties": "minimum validation NLL then earliest epoch"},
    }
    write_json(metadata_file, run_metadata)
    train_dataset = FieldDataset(train_records, config["data"], seed, fold, "train")
    validation_dataset = FieldDataset(validation_records, config["data"], seed, fold, "validation")
    validation_loader = make_loader(validation_dataset, config["training"])
    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_dataset.set_epoch(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loader = make_loader(train_dataset, config["training"], shuffle=True, epoch=epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, amp, config["training"].get("gradient_clip"))
        _, validation_metrics = predict_loader(
            model, validation_loader, device, prior, config["training"]["prior_strength"],
            config["data"]["name"], config["variant"], seed, fold,
        )
        scheduler.step()
        log_row = {"dataset": config["data"]["name"], "model": config["variant"],
                   "seed": int(seed), "fold": int(fold), "epoch": epoch + 1, "learning_rate": learning_rate}
        log_row.update({"train_" + name: value for name, value in train_metrics.items()})
        log_row.update({"validation_" + name: value for name, value in validation_metrics.items()})
        append_csv(history_file, log_row)
        current_key = selection_key(validation_metrics, config["training"], epoch)
        checkpoint = {
            "signature": signature, "config": config, "seed": int(seed), "fold": int(fold), "epoch": epoch,
            "model_state": model.state_dict(), "training_prior": prior, "split": split_ids,
            "records_sha256": records_fingerprint(records), "validation_metrics": validation_metrics,
            "data_sha256": data_sha256, "experiment_signature": run_signature,
            "model_size": run_metadata["model_size"], "environment": run_metadata["environment"],
        }
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            save_checkpoint(best_file, checkpoint)
        if config["training"].get("save_last", True):
            save_checkpoint(last_file, dict(
                checkpoint,
                optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), scaler_state=scaler.state_dict(),
                random_state=random_state(), best_key=best_key, best_epoch=best_epoch,
            ))
    best = load_checkpoint(best_file)
    model.load_state_dict(best["model_state"], strict=True)
    checkpoint_id = file_hash(best_file)
    validation_predictions, validation_metrics = predict_loader(
        model, validation_loader, device, prior, config["training"]["prior_strength"],
        config["data"]["name"], config["variant"], seed, fold, checkpoint_id,
    )
    threshold = select_threshold(
        validation_predictions["label"], validation_predictions["score"], config["training"]["threshold_rule"],
        config["training"]["minimum_sensitivity"],
    )
    test_dataset = FieldDataset(test_records, config["data"], seed, fold, "test")
    test_loader = make_loader(test_dataset, config["training"])
    test_predictions, test_metrics = predict_loader(
        model, test_loader, device, prior, config["training"]["prior_strength"],
        config["data"]["name"], config["variant"], seed, fold, checkpoint_id,
    )
    validation_predictions = finalize_predictions(validation_predictions, threshold)
    test_predictions = finalize_predictions(test_predictions, threshold)
    test_threshold_metrics = binary_metrics(test_predictions["label"], test_predictions["score"], threshold)
    best["validation_threshold"] = float(threshold)
    best["selected_epoch"] = best_epoch + 1
    save_checkpoint(best_file, best)
    checkpoint_id = file_hash(best_file)
    for name, predictions in (("validation", validation_predictions), ("test", test_predictions)):
        predictions["checkpoint"] = checkpoint_id
        write_csv(output_directory / f"{name}_predictions.csv", predictions)
    run_metadata.update(
        status="complete", selected_epoch=best_epoch + 1, validation_threshold=threshold,
        validation_metrics=validation_metrics, test_metrics=test_metrics, test_validation_threshold_metrics=test_threshold_metrics,
        checkpoint_sha256=checkpoint_id,
    )
    aggregation = config.get("evaluation", {}).get("patient_aggregation")
    if aggregation:
        validation_patients = aggregate_patients(validation_predictions, aggregation)
        test_patients = aggregate_patients(test_predictions, aggregation)
        patient_threshold = select_threshold(validation_patients["label"], validation_patients["score"], config["training"]["threshold_rule"], config["training"]["minimum_sensitivity"])
        write_csv(output_directory / "validation_patient_predictions.csv", validation_patients)
        write_csv(output_directory / "test_patient_predictions.csv", test_patients)
        run_metadata["patient_aggregation"] = aggregation
        run_metadata["patient_validation_threshold"] = patient_threshold
        run_metadata["patient_metrics"] = binary_metrics(test_patients["label"], test_patients["score"])
        run_metadata["patient_validation_threshold_metrics"] = binary_metrics(test_patients["label"], test_patients["score"], patient_threshold)
    finalize_integrity(output_directory, run_metadata)
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_predictions, test_metrics


def summarize_results(predictions, output_directory, expected_folds=5, reference_seed=None):
    output_directory = Path(output_directory)
    if predictions.empty:
        raise ValueError("No predictions are available to summarize")
    output_directory.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "model", "seed", "fold", "sample_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("Out-of-fold predictions contain duplicate sample keys")
    if predictions.groupby(["dataset", "model", "seed", "sample_id"])["fold"].nunique().max() != 1:
        raise ValueError("A sample appears in more than one test fold within a seed")
    fold_rows, threshold_rows, pooled_rows = [], [], []
    for key, group in predictions.groupby(["dataset", "model", "seed", "fold"], sort=True):
        identifiers = dict(zip(["dataset", "model", "seed", "fold"], key))
        fold_rows.append(dict(identifiers, **binary_metrics(group["label"], group["score"])))
        if "validation_threshold" in group:
            thresholds = group["validation_threshold"].unique()
            if len(thresholds) != 1:
                raise ValueError("A fold has multiple validation thresholds")
            threshold_rows.append(dict(identifiers, **binary_metrics(group["label"], group["score"], float(thresholds[0]))))
    for key, group in predictions.groupby(["dataset", "model", "seed"], sort=True):
        pooled_rows.append(dict(zip(["dataset", "model", "seed"], key), **binary_metrics(group["label"], group["score"])))
    fold_frame = pd.DataFrame(fold_rows)
    write_csv(output_directory / "oof_predictions.csv", predictions)
    write_csv(output_directory / "metrics_by_fold.csv", fold_frame)
    write_csv(output_directory / "metrics_validation_threshold.csv", pd.DataFrame(threshold_rows))
    write_csv(output_directory / "metrics_pooled_oof.csv", pd.DataFrame(pooled_rows))
    seed_rows = []
    for key, group in fold_frame.groupby(["dataset", "model", "seed"], sort=True):
        complete = set(group["fold"]) == set(range(1, expected_folds + 1))
        row = dict(zip(["dataset", "model", "seed"], key), folds=len(group), complete=complete)
        for metric in METRIC_NAMES:
            row[metric + "_fold_mean"] = float(group[metric].mean(skipna=False)) if complete else float("nan")
            row[metric + "_fold_sd"] = float(group[metric].std(ddof=1, skipna=False)) if complete else float("nan")
        seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    write_csv(output_directory / "metrics_by_seed.csv", seeds)
    combined = []
    for key, group in seeds.loc[seeds["complete"]].groupby(["dataset", "model"], sort=True):
        row = dict(zip(["dataset", "model"], key), seeds=len(group))
        for metric in METRIC_NAMES:
            values = group[metric + "_fold_mean"]
            row[metric + "_seed_mean"] = float(values.mean(skipna=False))
            row[metric + "_seed_sd"] = float(values.std(ddof=1, skipna=False)) if len(values) > 1 else float("nan")
        combined.append(row)
    write_csv(output_directory / "metrics_across_seeds.csv", pd.DataFrame(combined))
    if reference_seed is not None:
        reference = seeds.loc[seeds["seed"] == reference_seed]
        write_json(output_directory / "reference_run.json", {
            "reference_seed": reference_seed, "aggregation": "unweighted fold mean and sample standard deviation",
            "rows": reference.to_dict("records"),
        })
    return fold_frame


def run_cross_validation(records, config, seeds=None, folds=None, resume=False, overwrite=False):
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("At least one training epoch is required")
    aggregation = config.get("evaluation", {}).get("patient_aggregation")
    if aggregation not in {None, "mean", "max", "noisy_or"}:
        raise ValueError("Unknown patient aggregation rule")
    if aggregation and any(not record.patient_id or record.patient_label not in {0, 1} for record in records):
        raise ValueError("Patient evaluation requires actual patient IDs and independent patient references before training")
    assignments = prepare_splits(records, config)
    selected_seeds = config["experiment"]["seeds"] if seeds is None else seeds
    selected_folds = list(range(1, config["experiment"]["folds"] + 1)) if folds is None else folds
    if not selected_seeds or not selected_folds or len(set(selected_seeds)) != len(selected_seeds) or len(set(selected_folds)) != len(selected_folds):
        raise ValueError("Requested seeds and folds must be nonempty and unique")
    if not set(selected_seeds).issubset(set(config["experiment"]["seeds"])):
        raise ValueError("Requested seeds must be declared in the configuration")
    if not set(selected_folds).issubset(set(range(1, config["experiment"]["folds"] + 1))):
        raise ValueError("Invalid requested folds")
    slug = "".join(character if character.isalnum() or character in "._-" else "_" for character in config["variant"])
    output = Path(config["experiment"]["output"]) / slug
    identity = experiment_identity(records, assignments, config)
    pairs = {(int(seed), int(fold)) for seed in selected_seeds for fold in selected_folds}
    preflight_output(output, records, assignments, config, identity[0], pairs, resume, overwrite)
    write_experiment_inputs(output, records, assignments, config, *identity)
    for seed in selected_seeds:
        for fold in selected_folds:
            fit_fold(records, assignments, config, int(seed), int(fold), output / f"seed_{seed}" / f"fold_{fold}", resume, overwrite, identity)
    predictions = collect_completed(output, records, assignments, config, identity[0])
    summarize_results(predictions, output, config["experiment"]["folds"], config["experiment"].get("reference_seed"))
    mark_summary(output, config, identity[0], predictions)
    return output
