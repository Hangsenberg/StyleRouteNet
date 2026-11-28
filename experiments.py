import argparse
from pathlib import Path

from srnet.config import VARIANTS, load_config, resolve_variant
from srnet.data import load_records
from srnet.engine import run_cross_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "public.yaml"))
    parser.add_argument("--variants", nargs="+", default=["full", "backbone", "no_pscla", "no_bsser", "no_etc"])
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--split-file")
    parser.add_argument("--pretrained-path")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Each variant must be listed once")
    config = load_config(args.config)
    for value, section, key in ((args.source, "data", "source"), (args.manifest, "data", "manifest"),
                                (args.split_file, "experiment", "split_file"), (args.pretrained_path, "model", "pretrained_path")):
        if value:
            config[section][key] = str(Path(value).resolve())
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    if args.device:
        config["training"]["device"] = args.device
    if args.output:
        config["experiment"]["output"] = str(Path(args.output).resolve())
    settings_list = [resolve_variant(config, variant) for variant in args.variants]
    records = load_records(config["data"])
    for settings in settings_list:
        print(run_cross_validation(records, settings, args.seeds, args.folds, args.resume, args.overwrite))


if __name__ == "__main__":
    main()
