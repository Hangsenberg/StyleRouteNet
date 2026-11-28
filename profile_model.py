import argparse
import time
from pathlib import Path

import numpy as np
import torch
from thop import profile
from torch import nn

from srnet.config import load_config, resolve_variant
from srnet.engine import model_size
from srnet.models import build_model
from srnet.runtime import environment, get_device, seed_everything, write_json


class ProfileWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)["scaled_logits"]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "public.yaml"))
    parser.add_argument("--variant", default="full")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 1 or args.repetitions < 2:
        raise ValueError("Batch size and warmup must be positive, and repetitions must be at least two")
    config = resolve_variant(load_config(args.config), args.variant)
    seed_everything(2026, deterministic=True)
    device = get_device(args.device)
    model = build_model(config, pretrained=False).to(device).eval()
    dimensions = (args.batch_size, 3, config["data"]["image_size"], config["data"]["image_size"])
    images = torch.rand(dimensions, device=device)
    size = model_size(model)
    wrapper = ProfileWrapper(model).eval()
    with torch.inference_mode():
        macs, _ = profile(wrapper, inputs=(images,), verbose=False)
    timings = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            wrapper(images)
        synchronize(device)
        for _ in range(args.repetitions):
            synchronize(device)
            started = time.perf_counter()
            wrapper(images)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
    metadata = dict(size, variant=args.variant, input_shape=dimensions, dtype="float32", device=str(device),
                    macs_per_image=float(macs) / args.batch_size, flops_2macs_per_image=float(2 * macs) / args.batch_size,
                    operation_count_scope="THOP registered operators; functional elementwise operations and the custom state-update loop are not included",
                    latency_scope="model forward including tensor normalization, excluding file I/O, image resizing and device transfer",
                    batch_latency_ms_mean=float(np.mean(timings)), batch_latency_ms_sd=float(np.std(timings, ddof=1)),
                    batch_latency_ms_median=float(np.median(timings)), latency_ms_per_image=float(np.mean(timings)) / args.batch_size,
                    warmup=args.warmup, repetitions=args.repetitions, environment=environment())
    write_json(args.output, metadata)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
