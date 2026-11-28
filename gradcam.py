import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch.nn import functional as F

from srnet.data import FieldDataset, load_records
from srnet.inference import checkpoint_partition, evaluate_records, evaluation_data, restore_model
from srnet.losses import corrected_logits
from srnet.runtime import file_hash, write_json


def activation_map(model, images, style_images, prior, strength, target):
    with torch.enable_grad():
        output = model(images, style_images=style_images, return_features=True)
        features = output.get("feature_map")
        if features is None or features.ndim != 4:
            raise ValueError("Grad-CAM requires a spatial feature map from the selected model")
        logits = corrected_logits(output, prior, strength)
        gradients = torch.autograd.grad(logits[:, target].sum(), features, retain_graph=False)[0]
        if features.shape[-1] > features.shape[1] and features.shape[-1] > features.shape[2]:
            features = features.permute(0, 3, 1, 2)
            gradients = gradients.permute(0, 3, 1, 2)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else np.zeros_like(cam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--target", choices=["predicted", "reference", "positive"], default="predicted")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, checkpoint, config, device = restore_model(args.checkpoint, args.device)
    data = evaluation_data(config, source=args.source, manifest=args.manifest)
    records = checkpoint_partition(load_records(data), checkpoint, data=data)
    checkpoint_id = file_hash(args.checkpoint)
    predictions, _ = evaluate_records(model, checkpoint, config, data, records, device, checkpoint_id=checkpoint_id)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    if args.samples:
        if not set(args.samples).issubset(set(predictions["sample_id"])):
            raise ValueError("Activation maps can only use the checkpoint's held-out test images")
        chosen = predictions.loc[predictions["sample_id"].isin(args.samples)].sort_values("sample_id")
    else:
        if args.per_group < 1:
            raise ValueError("per-group must be positive")
        chosen = predictions.assign(correct=predictions["label"] == predictions["prediction"]).sort_values("sample_id").groupby(["label", "correct"], group_keys=False).head(args.per_group)
    dataset = FieldDataset(records, data, checkpoint["seed"], checkpoint["fold"], "test")
    indices = {record.sample_id: index for index, record in enumerate(records)}
    rows = []
    for row in chosen.itertuples(index=False):
        sample = dataset[indices[row.sample_id]]
        images = sample["image"].unsqueeze(0).to(device).requires_grad_(True)
        style_images = sample.get("style_image")
        if style_images is not None:
            style_images = style_images.unsqueeze(0).to(device)
        target = row.prediction if args.target == "predicted" else row.label if args.target == "reference" else 1
        cam = activation_map(model, images, style_images, checkpoint["training_prior"], config["training"]["prior_strength"], int(target))
        rgb = sample["image"].permute(1, 2, 0).numpy()
        heatmap = colormaps["jet"](cam)[..., :3]
        overlay = np.clip(0.6 * rgb + 0.4 * heatmap, 0, 1)
        stem = f"label_{row.label}_prediction_{row.prediction}_{row.sample_id}"
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output / f"{stem}_input.png")
        Image.fromarray(np.rint(heatmap * 255).astype(np.uint8)).save(output / f"{stem}_map.png")
        Image.fromarray(np.rint(overlay * 255).astype(np.uint8)).save(output / f"{stem}_overlay.png")
        np.save(output / f"{stem}_activation.npy", cam)
        rows.append({
            "sample_id": row.sample_id, "patient_id": row.patient_id, "label": row.label,
            "prediction": row.prediction, "score": row.score, "activation_target": int(target),
            "nonzero_activation": bool(cam.max() > 0), "pathologist_review": None,
        })
    write_json(output / "activation_maps.json", {"checkpoint_sha256": checkpoint_id, "selection": "specified sample IDs" if args.samples else "sample ID order within reference-class and correctness groups", "rows": rows})
    print(output.resolve())


if __name__ == "__main__":
    main()
