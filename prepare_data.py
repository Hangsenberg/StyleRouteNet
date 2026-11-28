import argparse
import hashlib
import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from srnet.config import load_config
from srnet.data import FieldDataset, dataset_fingerprint, load_records, records_fingerprint
from srnet.runtime import file_hash, write_json
from srnet.splits import prepare_splits


def inspect_images(records, config):
    dataset = FieldDataset(records, config["data"], 0, 0, "test")
    rows = []
    for record in records:
        content = dataset.read_bytes(record)
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            exif = image.getexif()
            detailed = exif.get_ifd(34665) if 34665 in exif else {}
            original_date = detailed.get(36867, exif.get(36867, ""))
            camera_date = exif.get(306, "")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        gray = rgb @ np.array([0.299, 0.587, 0.114])
        rows.append(dict(asdict(record), width=width, height=height, exif_datetime_original=str(original_date),
                         exif_datetime=str(camera_date), gray_p99=float(np.quantile(gray, 0.99)),
                         white_point_p99=float(np.mean(np.quantile(rgb, 0.99, axis=(0, 1))) * 255.0),
                         image_sha256=hashlib.sha256(content).hexdigest()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "public.yaml"))
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--split-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source:
        config["data"]["source"] = str(Path(args.source).resolve())
    if args.manifest:
        config["data"]["manifest"] = str(Path(args.manifest).resolve())
    if args.split_file:
        config["experiment"]["split_file"] = str(Path(args.split_file).resolve())
    records = load_records(config["data"])
    assignments = prepare_splits(records, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = inspect_images(records, config)
    frame.to_csv(output / "image_audit.csv", index=False)
    frame[["sample_id", "patient_id", "path", "label", "patient_label"]].to_csv(output / "manifest.csv", index=False)
    assignments.to_csv(output / "split_assignments.csv", index=False)
    summary = {
        "dataset": config["data"]["name"], "images": len(records), "records_sha256": records_fingerprint(records),
        "data_sha256": dataset_fingerprint(records, config["data"]),
        "field_label_counts": frame["label"].value_counts().sort_index().to_dict(),
        "archive_sha256": file_hash(config["data"]["source"]) if Path(config["data"]["source"]).is_file() else None,
        "duplicate_image_content": frame.loc[frame["image_sha256"].duplicated(keep=False), ["sample_id", "label", "image_sha256"]].to_dict("records"),
        "dimensions": frame.groupby(["width", "height"]).size().reset_index(name="images").to_dict("records"),
    }
    if frame["patient_id"].ne("").any():
        known = frame.loc[frame["patient_id"] != ""]
        patients = known.groupby("patient_id").agg(fields=("sample_id", "size"), minimum_field_label=("label", "min"), maximum_field_label=("label", "max"))
        patients.to_csv(output / "patient_field_composition.csv")
        summary.update(patients=int(len(patients)),
                       patients_with_only_positive_fields=int(patients["minimum_field_label"].eq(1).sum()),
                       patients_with_only_negative_fields=int(patients["maximum_field_label"].eq(0).sum()),
                       patients_with_mixed_field_labels=int(patients["minimum_field_label"].ne(patients["maximum_field_label"]).sum()))
    write_json(output / "dataset_summary.json", summary)
    print(output.resolve())


if __name__ == "__main__":
    main()
