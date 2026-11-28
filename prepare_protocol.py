"""Import retained fold IDs, never re-create folds from result magnitudes."""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

from srnet.data import IMAGE_SUFFIXES, ImageRecord, infer_label
from srnet.runtime import file_hash, write_csv, write_json
from srnet.splits import validate_assignments


PAPER_SEEDS = [73421, 18013, 92051, 46807, 31573]
REFERENCE_SEED = 73421


def import_protocol(oof_file, aliases_file, public_source, output, reference_file=None, overwrite=False):
    frame = pd.read_csv(oof_file, dtype=str, keep_default_na=False)
    required = {"dataset", "model", "sample_id", "patient_id", "fold", "seed", "label", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF data require {sorted(required)}")
    if set(frame["dataset"]) != {"public", "clinical"}:
        raise ValueError("Expected the public and clinical cohorts")
    if not frame["label"].isin(["0", "1"]).all() or not frame["fold"].isin(list("12345")).all():
        raise ValueError("Invalid labels or fold identifiers")
    frame[["seed", "fold", "label"]] = frame[["seed", "fold", "label"]].astype(int)
    if set(frame["seed"]) != set(PAPER_SEEDS):
        raise ValueError("The retained OOF seeds differ from the manuscript")
    if frame.duplicated(["dataset", "model", "seed", "sample_id"]).any():
        raise ValueError("Each seeded model must predict each sample once")
    for keys, values in [(["dataset", "seed", "sample_id"], ["fold", "label", "patient_id"]),
                         (["dataset", "sample_id"], ["label", "patient_id"])]:
        if frame.groupby(keys)[values].nunique().gt(1).any().any():
            raise ValueError("Models/seeds disagree on sample labels, patient IDs, or matched folds")
    if reference_file:
        reference = pd.read_csv(reference_file, dtype=str, keep_default_na=False)
        reference[["seed", "fold", "label"]] = reference[["seed", "fold", "label"]].astype(int)
        selected = frame.loc[frame["seed"].eq(REFERENCE_SEED) & frame["model"].isin(["SR-Net", "ConvNeXtV2"])]
        keys = ["dataset", "model", "sample_id"]
        left = reference[sorted(required)].sort_values(keys).reset_index(drop=True)
        right = selected[sorted(required)].sort_values(keys).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("The reference CSV is not the matching reference subset of the five-seed CSV")
    missing_public_ids = frame["dataset"].eq("public") & frame["patient_id"].eq("NA")
    frame.loc[missing_public_ids, "patient_id"] = ""
    aliases = pd.read_csv(aliases_file, dtype=str, keep_default_na=False)
    if aliases["sample_id"].duplicated().any():
        raise ValueError("Filename aliases must be unique")
    alias_table = aliases.copy()
    aliases = dict(zip(aliases["sample_id"], aliases["source_sample_id"]))
    source = Path(public_source)
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if Path(n).suffix.lower() in IMAGE_SUFFIXES and not n.startswith("__MACOSX/")]
    elif source.is_dir():
        names = [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    else:
        raise FileNotFoundError(source)
    by_name = {}
    for name in names:
        basename = Path(name).name
        if basename in by_name:
            raise ValueError(f"Ambiguous public filename: {basename}")
        by_name[basename] = name
    outputs, summaries = {"public_filename_aliases.csv": alias_table}, {}
    for cohort, count in (("public", 204), ("clinical", 576)):
        subset = frame.loc[frame["dataset"].eq(cohort)]
        models = {"SR-Net", "ConvNeXtV2", "hZACH-ViT" if cohort == "public" else "EfficientViM"}
        if set(subset["model"]) != models or not subset.groupby(["model", "seed"]).size().eq(count).all():
            raise ValueError(f"Incomplete model/seed coverage for {cohort}")
        reference = subset.loc[subset["model"].eq("SR-Net") & subset["seed"].eq(REFERENCE_SEED)]
        rows = []
        for row in reference.itertuples(index=False):
            path = ""
            if cohort == "public":
                basename = aliases.get(row.sample_id, row.sample_id)
                if basename not in by_name or infer_label(by_name[basename]) != row.label:
                    raise ValueError(f"Missing or mislabeled public source image: {row.sample_id}")
                path = by_name[basename]
            rows.append(dict(sample_id=row.sample_id, patient_id=row.patient_id, path=path, label=row.label, patient_label=""))
        manifest = pd.DataFrame(rows)
        split = subset[["sample_id", "patient_id", "label", "seed", "fold"]].drop_duplicates()
        records = [ImageRecord(r.sample_id, r.patient_id, r.path, r.label) for r in manifest.itertuples(index=False)]
        validate_assignments(split, records, PAPER_SEEDS, 5, cohort == "clinical")
        if cohort == "public" and (len(names) != count or set(manifest["path"]) != set(names)):
            raise ValueError("The public source and retained manifest are not identical image sets")
        outputs[f"{cohort}_splits.csv"] = split.sort_values(["seed", "sample_id"])
        outputs["public_manifest.csv" if cohort == "public" else "clinical_manifest_template.csv"] = manifest
        summaries[cohort] = dict(images=count, groups=int(manifest.loc[manifest["patient_id"].ne(""), "patient_id"].nunique()),
                                 reference_fold_supports=reference.groupby(["fold", "label"]).size().unstack().to_dict("index"))
    output = Path(output)
    if not overwrite and any((output / name).exists() for name in [*outputs, "provenance.json"]):
        raise FileExistsError("Protocol files already exist; inspect changes before using --overwrite")
    for name, table in outputs.items():
        write_csv(output / name, table)
    write_json(output / "provenance.json", {
        "origin": "Retained held-out assignments supplied by the user in the OOF CSVs",
        "seeds": PAPER_SEEDS, "reference_seed": REFERENCE_SEED,
        "scores_used_to_choose_or_order_folds": False,
        "validation_rule": "test_fold modulo five plus one; other three folds train, per manuscript",
        "inputs": {Path(oof_file).name: file_hash(oof_file), Path(aliases_file).name: file_hash(aliases_file),
                   **({Path(reference_file).name: file_hash(reference_file)} if reference_file else {})},
        "public_archive_sha256": file_hash(source) if source.is_file() else None,
        "outputs_sha256": {name: file_hash(output / name) for name in outputs},
        "clinical_image_paths": "Not supplied in prediction records; bind to actual images before training",
        "patient_reference_labels": "Not inferred from field labels; patient-level evaluation remains disabled",
        "cohorts": summaries,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--reference-oof")
    parser.add_argument("--aliases", required=True)
    parser.add_argument("--public-source", required=True)
    parser.add_argument("--output", default=str(Path(__file__).parent / "protocol"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(import_protocol(args.oof, args.aliases, args.public_source, args.output, args.reference_oof, args.overwrite))


if __name__ == "__main__":
    main()
