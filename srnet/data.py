import hashlib
import io
import math
import random
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import hed2rgb, rgb2hed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .runtime import file_hash, fingerprint, stable_seed


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    patient_id: str
    path: str
    label: int
    patient_label: int = -1


def records_fingerprint(records):
    return fingerprint([asdict(record) for record in sorted(records, key=lambda item: item.sample_id)])


def dataset_fingerprint(records, data_config):
    """Bind a run to actual image bytes, including clinical/extracted images."""
    source = Path(data_config["source"]).resolve()
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = set(archive.namelist())
        missing = [record.path for record in records if record.path not in members]
        if missing:
            raise FileNotFoundError(f"Images missing from the archive: {missing[:3]}")
        content = {"archive_sha256": file_hash(source)}
    else:
        content = {}
        for record in sorted(records, key=lambda item: item.sample_id):
            path = Path(record.path)
            if not path.is_absolute():
                path = source / path
            if not path.is_file():
                raise FileNotFoundError(f"Image path not found for {record.sample_id}: {path}")
            content[record.sample_id] = file_hash(path)
    return fingerprint({"records_sha256": records_fingerprint(records), "content": content})


def infer_label(path):
    names = {part.lower() for part in PurePosixPath(path.replace("\\", "/")).parts[:-1]}
    negative = bool(names & {"negative", "neg", "0"})
    positive = bool(names & {"positive", "pos", "1"})
    if negative == positive:
        raise ValueError(f"Cannot infer an unambiguous binary label from {path}")
    return int(positive)


def load_records(data_config):
    source = Path(data_config["source"]).resolve()
    expected_hash = data_config.get("archive_sha256")
    if expected_hash and source.is_file() and file_hash(source) != expected_hash:
        raise ValueError("The dataset archive SHA-256 does not match the configured release")
    manifest = data_config.get("manifest")
    records = []
    if manifest:
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        if not {"path", "label"}.issubset(frame.columns):
            raise ValueError("The manifest must contain path and label")
        for row in frame.to_dict("records"):
            relative = row["path"].replace("\\", "/")
            if not relative.strip():
                raise ValueError("Image path is empty. Bind the retained sample IDs to actual images before training; see bind_clinical.py.")
            sample_id = row.get("sample_id") or hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
            patient_label = int(row["patient_label"]) if row.get("patient_label") else -1
            records.append(ImageRecord(sample_id, row.get("patient_id", ""), relative, int(row["label"]), patient_label))
    else:
        if data_config.get("kind") == "clinical" or data_config.get("group_by_patient"):
            raise ValueError("Clinical data require a manifest with a de-identified patient_id for every field")
        if source.is_file() and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                names = sorted(name for name in archive.namelist() if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES and not name.startswith("__MACOSX/"))
        elif source.is_dir():
            names = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise FileNotFoundError(source)
        for name in names:
            sample_id = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
            records.append(ImageRecord(sample_id, "", name, infer_label(name)))
    if not records:
        raise ValueError("No image records were found")
    identifiers = [record.sample_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_id must be unique within a cohort")
    if len({record.path for record in records}) != len(records):
        raise ValueError("Each image path must have exactly one sample_id")
    if any(record.label not in (0, 1) for record in records):
        raise ValueError("Field labels must be binary")
    if data_config.get("kind") == "clinical" or data_config.get("group_by_patient"):
        if any(not record.patient_id for record in records):
            raise ValueError("Every clinical image requires a de-identified patient_id")
    patient_labels = {}
    for record in records:
        if record.patient_label not in (-1, 0, 1):
            raise ValueError("patient_label must be 0, 1, or empty")
        if record.patient_id and record.patient_label != -1:
            if record.patient_id in patient_labels and patient_labels[record.patient_id] != record.patient_label:
                raise ValueError(f"Inconsistent independent patient label: {record.patient_id}")
            patient_labels[record.patient_id] = record.patient_label
    return records


@contextmanager
def transform_seed(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class MacenkoNormalization:
    def __init__(self, alpha=1.0, beta=0.15):
        self.alpha = alpha
        self.beta = beta
        self.reference_stains = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float64)
        self.reference_concentrations = np.array([1.9705, 1.0308], dtype=np.float64)

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        optical_density = -np.log((rgb.reshape(-1, 3) + 1.0) / 256.0)
        tissue = optical_density[np.all(optical_density > self.beta, axis=1)]
        if tissue.shape[0] < 16:
            return image.convert("RGB")
        covariance = np.cov(tissue, rowvar=False)
        _, eigenvectors = np.linalg.eigh(covariance)
        plane = eigenvectors[:, -2:]
        plane = plane * np.where(plane.sum(axis=0) < 0, -1.0, 1.0)[None, :]
        projected = tissue @ plane
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        lower, upper = np.percentile(angles, [self.alpha, 100.0 - self.alpha])
        first = plane @ np.array([np.cos(lower), np.sin(lower)])
        second = plane @ np.array([np.cos(upper), np.sin(upper)])
        stains = np.stack((first, second), axis=1) if first[0] > second[0] else np.stack((second, first), axis=1)
        concentrations = np.linalg.lstsq(stains, optical_density.T, rcond=None)[0]
        upper_concentrations = np.percentile(concentrations, 99, axis=1)
        if not np.isfinite(upper_concentrations).all() or np.any(upper_concentrations <= 1e-8):
            return image.convert("RGB")
        normalized = concentrations * (self.reference_concentrations / upper_concentrations)[:, None]
        reconstructed = 256.0 * np.exp(-self.reference_stains @ normalized) - 1.0
        reconstructed = np.clip(reconstructed.T.reshape(rgb.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(reconstructed, mode="RGB")


class HEDAugmentation:
    def __init__(self, strength=0.05):
        self.strength = float(strength)

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        stains = rgb2hed(rgb)
        stains[..., :2] *= np.random.uniform(1.0 - self.strength, 1.0 + self.strength, size=2)
        augmented = np.clip(hed2rgb(stains), 0.0, 1.0)
        return Image.fromarray(np.rint(augmented * 255).astype(np.uint8), mode="RGB")


def build_transform(config, training):
    interpolation = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }[config.get("interpolation", "bilinear")]
    image_size = int(config["image_size"])
    steps = []
    if config.get("stain") == "macenko":
        steps.append(MacenkoNormalization())
    if training:
        steps.extend([
            T.RandomResizedCrop(image_size, scale=tuple(config["crop_scale"]), ratio=tuple(config["crop_ratio"]), interpolation=interpolation),
            T.RandomHorizontalFlip(config["horizontal_flip"]),
            T.RandomVerticalFlip(config["vertical_flip"]),
            T.RandomRotation(config["rotation"], interpolation=interpolation, fill=(255, 255, 255)),
            T.ColorJitter(config["brightness"], config["contrast"], config["saturation"], config["hue"]),
            T.RandomAutocontrast(config["autocontrast"]),
        ])
        if config.get("stain") == "hed":
            steps.append(T.RandomApply([HEDAugmentation(config["stain_strength"])], p=0.5))
    else:
        steps.append(T.Resize((image_size, image_size), interpolation=interpolation))
    steps.append(T.ToTensor())
    return T.Compose(steps)


def region_mask(image, ratio, seed, shape="square"):
    if not 0 <= ratio < 1:
        raise ValueError("The missing-area ratio must be in [0, 1)")
    if ratio == 0:
        return image, 0.0
    _, height, width = image.shape
    rng = np.random.default_rng(seed)
    target_area = ratio * height * width
    if shape == "square":
        side = min(height, width, max(1, round(math.sqrt(target_area))))
        top = int(rng.integers(0, height - side + 1))
        left = int(rng.integers(0, width - side + 1))
        mask = torch.zeros((height, width), dtype=torch.bool)
        mask[top:top + side, left:left + side] = True
    elif shape == "rectangle":
        lower = max(0.5, target_area / height**2)
        upper = min(2.0, width**2 / target_area)
        aspect = math.exp(rng.uniform(math.log(lower), math.log(upper)))
        mask_height = min(height, max(1, round(math.sqrt(target_area / aspect))))
        mask_width = min(width, max(1, round(target_area / mask_height)))
        top = int(rng.integers(0, height - mask_height + 1))
        left = int(rng.integers(0, width - mask_width + 1))
        mask = torch.zeros((height, width), dtype=torch.bool)
        mask[top:top + mask_height, left:left + mask_width] = True
    elif shape == "ellipse":
        radius = math.sqrt(ratio / math.pi)
        center_y = rng.uniform(radius, 1.0 - radius) if radius < 0.5 else 0.5
        center_x = rng.uniform(radius, 1.0 - radius) if radius < 0.5 else 0.5
        y = (torch.arange(height) + 0.5) / height
        x = (torch.arange(width) + 0.5) / width
        mask = ((y[:, None] - center_y).square() + (x[None, :] - center_x).square()) <= radius**2
    else:
        raise ValueError(f"Unknown mask shape: {shape}")
    result = image.clone()
    result[:, mask] = image.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    return result, float(mask.float().mean())


class FieldDataset(Dataset):
    def __init__(self, records, data_config, seed, fold, split, mask_ratio=0.0, mask_shape="square", permute_styles=None):
        self.records = list(records)
        self.config = dict(data_config)
        self.source = Path(data_config["source"]).resolve()
        self.is_archive = self.source.is_file() and zipfile.is_zipfile(self.source)
        self.archive = None
        self.seed = int(seed)
        self.fold = int(fold)
        self.split = split
        self.training = split == "train"
        self.transform = build_transform(data_config, self.training)
        self.mask_ratio = float(mask_ratio)
        self.mask_shape = mask_shape
        enabled = data_config.get("permute_styles", False)
        stage = data_config.get("permutation_stage", "test")
        self.permute_styles = enabled and (stage == "all" or stage == split or stage == "evaluation" and split in {"validation", "test"})
        if permute_styles is not None:
            self.permute_styles = bool(permute_styles)
        self.set_epoch(0)

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["archive"] = None
        return state

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        generator = np.random.default_rng(stable_seed(self.seed, self.fold, self.split, self.epoch, "style"))
        self.style_indices = generator.permutation(len(self.records))

    def read_image(self, record):
        with Image.open(io.BytesIO(self.read_bytes(record))) as image:
            return image.convert("RGB").copy()

    def read_bytes(self, record):
        if self.is_archive:
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.source)
            return self.archive.read(record.path)
        path = Path(record.path)
        if not path.is_absolute():
            path = self.source / path
        return path.read_bytes()

    def view(self, record, index):
        seed = stable_seed(self.seed, self.fold, self.split, self.epoch if self.training else 0, record.sample_id, index)
        with transform_seed(seed):
            image = self.transform(self.read_image(record))
        mask_seed = stable_seed(self.config.get("mask_seed", 1729), record.sample_id, self.mask_ratio, self.mask_shape)
        return region_mask(image, self.mask_ratio, mask_seed, self.mask_shape)

    def __getitem__(self, index):
        record = self.records[index]
        first, actual_ratio = self.view(record, 0)
        result = {
            "image": first,
            "label": record.label,
            "patient_label": record.patient_label,
            "sample_id": record.sample_id,
            "patient_id": record.patient_id,
            "mask_fraction": actual_ratio,
            "style_sample_id": record.sample_id,
        }
        if self.training:
            result["image_second"] = self.view(record, 1)[0]
        if self.permute_styles:
            donor = self.records[int(self.style_indices[index])]
            result["style_image"] = self.view(donor, 0)[0]
            result["style_sample_id"] = donor.sample_id
            if self.training:
                result["style_image_second"] = self.view(donor, 1)[0]
        return result


def make_loader(dataset, training_config, shuffle=False, epoch=0):
    generator = torch.Generator()
    generator.manual_seed(stable_seed(dataset.seed, dataset.fold, dataset.split, epoch, "loader"))
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=False,
        generator=generator,
    )
