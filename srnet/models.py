import importlib
from copy import deepcopy
from pathlib import Path

import timm
import torch
from torch import nn


def style_statistics(images, eps=1e-6):
    values = images.float().flatten(2)
    mean = values.mean(dim=-1)
    centered = values - mean.unsqueeze(-1)
    variance = centered.square().mean(dim=-1)
    standard_deviation = variance.sqrt()
    skewness = centered.pow(3).mean(dim=-1) / (variance + eps).pow(1.5)
    return torch.cat((mean, standard_deviation, skewness), dim=-1)


class RGBNormalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images):
        return (images - self.mean) / self.std


class PSCLA(nn.Module):
    def __init__(self, channels, rank=64, norm_eps=1e-5, gate="conditioned", constant_gate=0.5):
        super().__init__()
        self.gate_mode = gate
        self.constant_gate = float(constant_gate)
        self.style_encoder = nn.Sequential(
            nn.LayerNorm(9, eps=norm_eps),
            nn.Linear(9, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
            nn.Sigmoid(),
        )
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(()))
        if gate not in {"constant", "conditioned"}:
            raise ValueError(f"Unknown gate: {gate}")

    def forward(self, features, style):
        hidden = self.activation(self.down(features))
        if self.gate_mode == "constant":
            gate = hidden.new_full((hidden.shape[0], hidden.shape[1], 1, 1), self.constant_gate)
        else:
            gate = self.style_encoder(style).unsqueeze(-1).unsqueeze(-1)
        return features + self.alpha * self.up(hidden * gate)


class DirectionalRecurrence(nn.Module):
    def __init__(self, channels, norm_eps=1e-5, decay_eps=1e-4):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=norm_eps)
        self.content = nn.Linear(channels, channels, bias=False)
        self.gate = nn.Linear(channels, channels, bias=False)
        self.decay_logits = nn.Parameter(torch.zeros(channels))
        self.input_logits = nn.Parameter(torch.zeros(channels))
        self.decay_eps = decay_eps

    def forward(self, tokens):
        normalized = self.norm(tokens)
        content = self.content(normalized).tanh().float()
        gates = self.gate(normalized).sigmoid().float()
        decay = (1.0 - self.decay_eps) * self.decay_logits.float().sigmoid()
        amplitude = self.input_logits.float().sigmoid()
        updates = amplitude * gates * content
        state = updates.new_zeros(updates.shape[0], updates.shape[2])
        states = []
        for index in range(updates.shape[1]):
            state = decay * state + updates[:, index]
            states.append(state)
        return torch.stack(states, dim=1)


class BSSER(nn.Module):
    def __init__(self, channels, rank=64, norm_eps=1e-5, decay_eps=1e-4, scan_order="serpentine", bidirectional=True):
        super().__init__()
        if scan_order not in {"serpentine", "raster"}:
            raise ValueError(f"Unknown scan order: {scan_order}")
        self.scan_order = scan_order
        self.bidirectional = bidirectional
        self.projection = nn.Linear(channels, channels, bias=False)
        self.forward_scan = DirectionalRecurrence(channels, norm_eps, decay_eps)
        self.backward_scan = DirectionalRecurrence(channels, norm_eps, decay_eps) if bidirectional else None
        self.output_projection = nn.Linear(channels, channels, bias=False)
        self.evidence_gate = nn.Sequential(
            nn.LayerNorm(channels, eps=norm_eps),
            nn.Linear(channels, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, channels, bias=False),
            nn.Sigmoid(),
        )
        self.mlp_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.GELU(),
            nn.Linear(channels, channels, bias=False),
        )
        self.gamma_r = nn.Parameter(torch.ones(()))
        self.gamma_m = nn.Parameter(torch.ones(()))

    def permutation(self, height, width, device):
        grid = torch.arange(height * width, device=device).view(height, width)
        if self.scan_order == "raster":
            order = grid.flatten()
        else:
            order = torch.cat([row if index % 2 == 0 else row.flip(0) for index, row in enumerate(grid)])
        return order, order.argsort()

    def forward(self, features):
        batch, channels, height, width = features.shape
        tokens = self.projection(features.flatten(2).transpose(1, 2))
        order, inverse = self.permutation(height, width, tokens.device)
        sequence = tokens.index_select(1, order)
        routed = self.forward_scan(sequence)
        if self.backward_scan is not None:
            backward = self.backward_scan(sequence.flip(1)).flip(1)
            routed = (routed + backward) * 0.5
        routed = self.output_projection(routed).index_select(1, inverse)
        evidence = self.evidence_gate(routed)
        combined = tokens + self.gamma_r * evidence * routed
        combined = combined + self.gamma_m * self.mlp(self.mlp_norm(combined))
        return combined.transpose(1, 2).reshape(batch, channels, height, width)


class AttentionRouting(nn.Module):
    def __init__(self, channels, target_parameters, heads=8, norm_eps=1e-5):
        super().__init__()
        if channels % heads:
            raise ValueError("Attention heads must divide the feature dimension")
        fixed_parameters = 4 * channels * channels + 4 * channels + 2
        hidden = max(1, round((target_parameters - fixed_parameters) / (2 * channels)))
        self.attention_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.0, bias=False, batch_first=True)
        self.mlp_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
        )
        self.gamma_r = nn.Parameter(torch.ones(()))
        self.gamma_m = nn.Parameter(torch.ones(()))
        self.target_parameters = int(target_parameters)
        self.hidden_dimension = hidden

    def forward(self, features):
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        normalized = self.attention_norm(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.gamma_r * attended
        tokens = tokens + self.gamma_m * self.mlp(self.mlp_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class ETCHead(nn.Module):
    def __init__(self, channels=768, norm_eps=1e-5, temperature=True, temperature_min=0.7, temperature_max=2.5):
        super().__init__()
        if not 0 < temperature_min < temperature_max:
            raise ValueError("Invalid temperature bounds")
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels, eps=norm_eps),
            nn.Linear(channels, channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(channels // 2, 2, bias=False),
        )
        self.temperature_branch = None
        if temperature:
            self.temperature_branch = nn.Sequential(
                nn.LayerNorm(channels, eps=norm_eps),
                nn.Linear(channels, channels // 4, bias=False),
                nn.GELU(),
                nn.Linear(channels // 4, 1, bias=False),
            )
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def forward(self, embedding):
        logits = self.classifier(embedding)
        if self.temperature_branch is None:
            temperature = logits.new_ones(logits.shape[0], 1)
        else:
            response = self.temperature_branch(embedding).sigmoid()
            temperature = self.temperature_min + (self.temperature_max - self.temperature_min) * response
        return logits, temperature


def load_backbone_weights(backbone, filename):
    checkpoint = torch.load(Path(filename), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        if not key.startswith(("head.", "classifier.", "fc.")):
            normalized[key] = value
    missing, unexpected = backbone.load_state_dict(normalized, strict=False)
    invalid_missing = [key for key in missing if not key.startswith(("head.", "classifier.", "fc."))]
    if invalid_missing or unexpected:
        raise ValueError(f"Incompatible backbone weights: missing={invalid_missing}, unexpected={unexpected}")


class StyleRouteNet(nn.Module):
    def __init__(self, model_config, data_config, pretrained=None):
        super().__init__()
        config = deepcopy(model_config)
        use_pretrained = config.get("pretrained", True) if pretrained is None else pretrained
        if config.get("pretrained_path"):
            use_pretrained = False
        self.normalization = RGBNormalization(data_config["mean"], data_config["std"])
        self.backbone = timm.create_model(
            config["backbone"],
            pretrained=use_pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=0.0,
            drop_path_rate=config.get("drop_path_rate", 0.0),
        )
        if config.get("pretrained_path") and pretrained is not False:
            load_backbone_weights(self.backbone, config["pretrained_path"])
        self.backbone.head = nn.Identity()
        channels = self.backbone.num_features
        rank = config.get("rank", 64)
        eps = config.get("layer_norm_eps", 1e-5)
        self.style_eps = config.get("style_eps", 1e-6)
        self.adapter = None
        if config.get("adapter", True):
            self.adapter = PSCLA(channels, rank, eps, config.get("gate", "conditioned"), config.get("constant_gate", 0.5))
        self.routing = None
        if config.get("routing", True):
            if config.get("routing_type", "recurrent") == "attention":
                reference = BSSER(channels, rank, eps, config.get("decay_eps", 1e-4))
                target = sum(parameter.numel() for parameter in reference.parameters())
                self.routing = AttentionRouting(channels, target, config.get("attention_heads", 8), eps)
            else:
                self.routing = BSSER(
                    channels, rank, eps, config.get("decay_eps", 1e-4),
                    config.get("scan_order", "serpentine"), config.get("bidirectional", True),
                )
        self.output_norm = nn.BatchNorm2d(channels)
        self.head = ETCHead(
            channels, eps, config.get("temperature", True),
            config.get("temperature_min", 0.7), config.get("temperature_max", 2.5),
        )

    def forward(self, images, style_images=None, return_features=False):
        features = self.backbone.forward_features(self.normalization(images))
        if self.adapter is not None:
            statistics = style_statistics(images if style_images is None else style_images, self.style_eps)
            features = self.adapter(features, statistics)
        if self.routing is not None:
            features = self.routing(features)
        normalized = self.output_norm(features)
        embedding = normalized.mean(dim=(2, 3))
        logits, temperature = self.head(embedding)
        output = {"logits": logits, "temperature": temperature, "scaled_logits": logits.float() / temperature.float()}
        if return_features:
            output.update(feature_map=features, embedding=embedding)
        return output


class TimmClassifier(nn.Module):
    def __init__(self, model_name, model_config, data_config, pretrained=None):
        super().__init__()
        use_pretrained = model_config.get("pretrained", True) if pretrained is None else pretrained
        if model_config.get("pretrained_path"):
            use_pretrained = False
        self.normalization = RGBNormalization(data_config["mean"], data_config["std"])
        self.network = timm.create_model(model_name, pretrained=use_pretrained, num_classes=2, drop_rate=0.0)
        if model_config.get("pretrained_path") and pretrained is not False:
            load_backbone_weights(self.network, model_config["pretrained_path"])

    def forward(self, images, style_images=None, return_features=False):
        if return_features:
            features = self.network.forward_features(self.normalization(images))
            logits = self.network.forward_head(features)
        else:
            logits = self.network(self.normalization(images))
        temperature = logits.new_ones(logits.shape[0], 1)
        output = {"logits": logits, "temperature": temperature, "scaled_logits": logits.float()}
        if return_features:
            output["feature_map"] = features
        return output


def build_model(config, pretrained=None):
    variant = config.get("variant", "full")
    if variant == "backbone":
        return TimmClassifier(config["model"]["backbone"], config["model"], config["data"], pretrained)
    if variant.startswith("timm:"):
        return TimmClassifier(variant.split(":", 1)[1], config["model"], config["data"], pretrained)
    if variant.startswith("factory:"):
        specification = variant.split(":", 1)[1]
        module_name, function_name = specification.rsplit(".", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        return factory(deepcopy(config), pretrained=pretrained)
    return StyleRouteNet(config["model"], config["data"], pretrained)
