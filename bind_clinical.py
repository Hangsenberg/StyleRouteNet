"""Join actual image paths to retained clinical IDs; never infer patient labels."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from srnet.data import dataset_fingerprint, load_records
from srnet.runtime import file_hash, write_csv, write_json


def bind_images(template, mapping, source, output, overwrite=False):
    reference = pd.read_csv(template, dtype=str, keep_default_na=False)
    paths = pd.read_csv(mapping, dtype=str, keep_default_na=False)
    if not {"sample_id", "path"}.issubset(paths) or paths["sample_id"].duplicated().any():
        raise ValueError("Mapping requires a unique sample_id and path for each image")
    if reference["sample_id"].duplicated().any() or set(paths["sample_id"]) != set(reference["sample_id"]):
        raise ValueError("Actual clinical mapping must cover the retained sample IDs exactly once")
    joined = reference.drop(columns="path").merge(paths[["sample_id", "path"]], on="sample_id", validate="one_to_one", sort=False)
    if joined["path"].str.strip().eq("").any():
        raise ValueError("Every clinical sample requires its actual image path")
    resolved = []
    for row in joined.itertuples(index=False):
        path = Path(row.path)
        path = path if path.is_absolute() else Path(source) / path
        resolved.append(str(path.resolve()))
        with Image.open(path) as image:
            image.verify()
    if len(set(resolved)) != len(resolved):
        raise ValueError("Two sample IDs refer to the same actual image file")
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    temporary = output.with_suffix(".binding.csv")
    write_csv(temporary, joined)
    try:
        data = dict(source=str(Path(source).resolve()), manifest=str(temporary), kind="clinical", group_by_patient=True)
        digest = dataset_fingerprint(load_records(data), data)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_csv(output, joined)
    write_json(output.with_suffix(".json"), dict(template_sha256=file_hash(template), mapping_sha256=file_hash(mapping),
                                               data_sha256=digest, independent_patient_labels_inferred=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, help="CSV with retained sample_id and actual path")
    parser.add_argument("--source", required=True)
    parser.add_argument("--template", default=str(Path(__file__).parent / "protocol/clinical_manifest_template.csv"))
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/clinical.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(bind_images(args.template, args.mapping, args.source, args.output, args.overwrite))


if __name__ == "__main__":
    main()
