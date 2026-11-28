import argparse
from pathlib import Path

from srnet.config import VARIANTS, load_config, resolve_variant
from srnet.data import load_records
from srnet.engine import run_cross_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "public.yaml"))
    parser.add_argument("--variant", default="full", help=", ".join(VARIANTS) + ", timm:MODEL, factory:MODULE.FUNCTION")
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--split-file")
    parser.add_argument("--output")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--pretrained-path")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source:
        config["data"]["source"] = str(Path(args.source).resolve())
    if args.manifest:
        config["data"]["manifest"] = str(Path(args.manifest).resolve())
    if args.split_file:
        config["experiment"]["split_file"] = str(Path(args.split_file).resolve())
    if args.output:
        config["experiment"]["output"] = str(Path(args.output).resolve())
    if args.device:
        config["training"]["device"] = args.device
    if args.pretrained_path:
        config["model"]["pretrained_path"] = str(Path(args.pretrained_path).resolve())
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    config = resolve_variant(config, args.variant)
    records = load_records(config["data"])
    output = run_cross_validation(records, config, args.seeds, args.folds, args.resume, args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
