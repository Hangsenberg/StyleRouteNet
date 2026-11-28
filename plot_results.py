import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from srnet.inference import read_predictions
from srnet.metrics import calibration_bins


def save_figure(figure, destination):
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_predictions(predictions, output):
    for (dataset, seed), subset in predictions.groupby(["dataset", "seed"], sort=True):
        folder = output / str(dataset) / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        roc_figure, roc_axis = plt.subplots(figsize=(5.2, 4.5))
        pr_figure, pr_axis = plt.subplots(figsize=(5.2, 4.5))
        calibration_figure, calibration_axis = plt.subplots(figsize=(5.2, 4.5))
        score_figure, score_axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
        fold_rows = []
        for model, group in subset.groupby("model", sort=True):
            if group["sample_id"].duplicated().any():
                raise ValueError("A sample has repeated predictions within the same seeded model")
            fpr, tpr, _ = roc_curve(group["label"], group["score"])
            precision, recall, _ = precision_recall_curve(group["label"], group["score"])
            roc_axis.plot(fpr, tpr, label=f"{model} ({auc(fpr, tpr):.3f})")
            pr_axis.plot(recall, precision, label=f"{model} ({auc(recall, precision):.3f})")
            calibration = calibration_bins(group["label"], group["score"])
            calibration_axis.plot(calibration["confidence"], calibration["accuracy"], marker="o", label=model)
            for label in (0, 1):
                score_axes[label].hist(group.loc[group["label"] == label, "score"], bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.5, label=model)
            for fold, fold_group in group.groupby("fold", sort=True):
                fold_fpr, fold_tpr, _ = roc_curve(fold_group["label"], fold_group["score"])
                fold_rows.append({"model": model, "fold": int(fold), "auroc": auc(fold_fpr, fold_tpr)})
        roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        roc_axis.legend(fontsize=8)
        pr_axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        pr_axis.legend(fontsize=8)
        calibration_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        calibration_axis.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
        calibration_axis.legend(fontsize=8)
        for label in (0, 1):
            score_axes[label].axvline(0.5, color="gray", linestyle="--", linewidth=1)
            score_axes[label].set(xlabel="Positive-class score", title="Negative reference" if label == 0 else "Positive reference", xlim=(0, 1))
            score_axes[label].legend(fontsize=8)
        score_axes[0].set_ylabel("Image count")
        for name, figure in (("pooled_roc", roc_figure), ("pooled_precision_recall", pr_figure), ("reliability", calibration_figure), ("score_distributions", score_figure)):
            save_figure(figure, folder / name)
        folds = pd.DataFrame(fold_rows)
        fold_figure, fold_axis = plt.subplots(figsize=(max(5.2, folds["model"].nunique() * 1.2), 4.5))
        for index, (model, group) in enumerate(folds.groupby("model", sort=True)):
            fold_axis.scatter(np.full(len(group), index) + np.linspace(-0.12, 0.12, len(group)), group["auroc"], s=26)
            fold_axis.errorbar(index, group["auroc"].mean(), yerr=group["auroc"].std(ddof=1), fmt="s", color="black", capsize=4)
        fold_axis.set_xticks(range(folds["model"].nunique()), sorted(folds["model"].unique()), rotation=35, ha="right")
        fold_axis.set(ylabel="Fold AUROC", ylim=(0, 1))
        folds.to_csv(folder / "plotted_fold_auc.csv", index=False)
        save_figure(fold_figure, folder / "fold_auc")


def _history_identity(path, history):
    names = ("dataset", "model", "seed", "fold")
    if set(names).issubset(history.columns):
        if history[list(names)].drop_duplicates().shape[0] != 1:
            raise ValueError(f"History contains multiple run identities: {path}")
        row = history.iloc[0]
        return str(row["dataset"]), str(row["model"]), int(row["seed"]), int(row["fold"])
    metadata_file = path.parent / "run.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        config = metadata["config"]
        return config["data"]["name"], config["variant"], int(metadata["seed"]), int(metadata["fold"])
    raise ValueError(f"History requires dataset/model/seed/fold columns or an adjacent run.json: {path}")


def plot_histories(filenames, output, seed=None):
    plans, destinations = [], set()
    for filename in filenames:
        path = Path(filename)
        history = pd.read_csv(path)
        dataset, model, run_seed, fold = _history_identity(path, history)
        if seed is not None and run_seed != seed:
            continue
        slug = lambda value: "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))
        destination = output / "learning_curves" / slug(dataset) / slug(model) / f"seed_{run_seed}" / f"fold_{fold}"
        if destination in destinations:
            raise ValueError(f"Two histories would overwrite the same plot: {destination}")
        destinations.add(destination)
        plans.append((history, destination, dataset, model, run_seed, fold))
    for history, destination, dataset, model, run_seed, fold in plans:
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
        axes[0].plot(history["epoch"], history["train_loss"], label="Training objective")
        axes[0].plot(history["epoch"], history["validation_loss"], label="Validation NLL")
        axes[0].set(xlabel="Epoch", ylabel="Loss")
        axes[0].legend(fontsize=8)
        axes[1].plot(history["epoch"], history["validation_auroc"], label="Validation AUROC")
        axes[1].set(xlabel="Epoch", ylabel="AUROC", ylim=(0, 1))
        axes[1].legend(fontsize=8)
        figure.suptitle(f"{dataset} / {model} / seed {run_seed} / fold {fold}")
        destination.mkdir(parents=True, exist_ok=True)
        save_figure(figure, destination / "training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="*", default=[])
    parser.add_argument("--histories", nargs="*", default=[])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.predictions and not args.histories:
        raise ValueError("At least one prediction file or training history is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.predictions:
        predictions = pd.concat([read_predictions(filename) for filename in args.predictions], ignore_index=True)
        predictions = predictions.loc[predictions["seed"] == args.seed].copy()
        if predictions.empty:
            raise ValueError("No predictions exist for the requested seed")
        plot_predictions(predictions, output)
    plot_histories(args.histories, output, args.seed)
    print(output.resolve())


if __name__ == "__main__":
    main()
