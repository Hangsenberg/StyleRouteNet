import torch
from torch import nn
from torch.nn import functional as F


class TwoViewObjective(nn.Module):
    def __init__(self, label_smoothing=0.05, consistency_weight=0.35, temperature_weight=0.01):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.consistency_weight = consistency_weight
        self.temperature_weight = temperature_weight

    def forward(self, first, second, labels):
        first_logits = first["scaled_logits"].float()
        second_logits = second["scaled_logits"].float()
        first_log = F.log_softmax(first_logits, dim=-1)
        second_log = F.log_softmax(second_logits, dim=-1)
        first_probability = first_log.exp()
        second_probability = second_log.exp()
        classification = 0.5 * (
            F.cross_entropy(first_logits, labels, label_smoothing=self.label_smoothing)
            + F.cross_entropy(second_logits, labels, label_smoothing=self.label_smoothing)
        )
        divergence = 0.5 * (
            first_probability * (first_log - second_log)
            + second_probability * (second_log - first_log)
        ).sum(dim=-1)
        confidence = 0.5 * (first_probability.max(dim=-1).values + second_probability.max(dim=-1).values)
        first_temperature = first["temperature"].float().flatten()
        second_temperature = second["temperature"].float().flatten()
        temperature_mean = 0.5 * (first_temperature + second_temperature)
        weights = confidence / (1.0 + temperature_mean)
        consistency = (weights * divergence).mean()
        temperature = 0.5 * ((first_temperature - 1.0).square().mean() + (second_temperature - 1.0).square().mean())
        loss = classification + self.consistency_weight * consistency + self.temperature_weight * temperature
        return {"loss": loss, "classification": classification, "consistency": consistency, "temperature": temperature}


def corrected_logits(output, prior, strength=0.05):
    logits = output["scaled_logits"].float()
    if strength == 0:
        return logits
    prior = torch.as_tensor(prior, dtype=logits.dtype, device=logits.device)
    if prior.shape != (2,) or torch.any(prior <= 0):
        raise ValueError("Both classes must occur in the training allocation")
    return logits - strength * prior.log().unsqueeze(0)
