from copy import deepcopy
from pathlib import Path

import yaml


VARIANTS = (
    "full",
    "backbone",
    "no_pscla",
    "no_bsser",
    "no_etc",
    "constant_gate",
    "raster",
    "unidirectional",
    "attention",
    "macenko",
    "stain_augmentation",
    "style_permutation",
)


def merge_config(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(filename):
    filename = Path(filename).resolve()
    with filename.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.pop("extends", None)
    if parent:
        config = merge_config(load_config(filename.parent / parent), config)
    config["config_file"] = str(filename)
    for section, keys in {
        "experiment": ("output", "split_file"),
        "data": ("source", "manifest"),
        "model": ("pretrained_path",),
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    config[section][key] = str((filename.parent / path).resolve())
    return config


def resolve_variant(config, variant):
    if variant not in VARIANTS and not variant.startswith(("timm:", "factory:")):
        raise ValueError(f"Unknown variant: {variant}")
    result = deepcopy(config)
    analysis = result["experiment"].get("analysis", "main")
    if result["experiment"].get("split_mode", "replay") == "replay":
        required_analysis = {"raster": "scan", "unidirectional": "scan",
                             "macenko": "appearance", "style_permutation": "appearance"}.get(variant)
        if required_analysis and analysis != required_analysis:
            raise ValueError(f"The manuscript's {variant} comparison requires analysis: {required_analysis} and its retained split file; use the matching analysis config.")
    result["variant"] = variant
    settings = result["model"]
    settings.update(
        adapter=True,
        routing=True,
        temperature=True,
        gate="conditioned",
        scan_order="serpentine",
        bidirectional=True,
        routing_type="recurrent",
    )
    result["data"]["permute_styles"] = False
    if variant == "no_pscla":
        settings["adapter"] = False
    elif variant == "no_bsser":
        settings["routing"] = False
    elif variant == "no_etc":
        settings["temperature"] = False
    elif variant == "constant_gate":
        settings["gate"] = "constant"
        settings["constant_gate"] = 1.0
    elif variant == "raster":
        settings["scan_order"] = "raster"
    elif variant == "unidirectional":
        settings["bidirectional"] = False
    elif variant == "attention":
        settings["routing_type"] = "attention"
    elif variant == "macenko":
        settings["adapter"] = False
        result["data"]["stain"] = "macenko"
    elif variant == "stain_augmentation":
        result["data"]["stain"] = "hed"
    elif variant == "style_permutation":
        result["data"]["permute_styles"] = True
        result["data"]["permutation_stage"] = "train"
    return result
