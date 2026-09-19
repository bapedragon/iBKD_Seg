"""Deterministic iBKD CBAM execution for the Cityscapes CUDA protocol."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CANDIDATE_ID = "flatmax_cpu_deform_v1"


def _conv2d_with_module(inputs: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
    return F.conv2d(
        inputs,
        module.weight,
        module.bias,
        module.stride,
        module.padding,
        module.dilation,
        module.groups,
    )


def _cpu_deformable_spatial(spatial: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run the modulated deformable-convolution equation on the CPU."""
    import torchvision

    target_device = inputs.device
    cpu_inputs = inputs.to("cpu")

    def cpu_conv(module: nn.Conv2d) -> torch.Tensor:
        return F.conv2d(
            cpu_inputs,
            module.weight.to("cpu"),
            None if module.bias is None else module.bias.to("cpu"),
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )

    offset = cpu_conv(spatial.offset)
    mask = 2.0 * torch.sigmoid(cpu_conv(spatial.modulator))
    output = torchvision.ops.deform_conv2d(
        input=cpu_inputs,
        offset=offset,
        weight=spatial.regular.weight.to("cpu"),
        bias=None if spatial.regular.bias is None else spatial.regular.bias.to("cpu"),
        stride=spatial.regular.stride,
        padding=spatial.padding,
        dilation=spatial.regular.dilation,
        mask=mask,
    )
    return output.to(target_device)


class DiagnosticCBAM(nn.Module):
    """Select original or deterministic-equivalent CBAM components."""

    def __init__(self, original: nn.Module, *, channel_mode: str, spatial_mode: str) -> None:
        super().__init__()
        if channel_mode not in {"identity", "adaptive", "flatmax"}:
            raise ValueError(channel_mode)
        if spatial_mode not in {"identity", "regular", "gpu_deform", "cpu_deform"}:
            raise ValueError(spatial_mode)
        self.original = original
        self.channel_mode = channel_mode
        self.spatial_mode = spatial_mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.channel_mode == "identity":
            return inputs
        channel = self.original.channel
        average = channel.average(inputs)
        if self.channel_mode == "adaptive":
            maximum = channel.maximum(inputs)
        else:
            # Flattened torch.max preserves AdaptiveMaxPool2d(1)'s forward value
            # and first-index tie gradient without its CUDA backward kernel.
            maximum = inputs.flatten(2).max(dim=2).values[:, :, None, None]
        outputs = inputs * torch.sigmoid(channel.mlp(average) + channel.mlp(maximum))
        pooled = torch.cat(
            (outputs.mean(dim=1, keepdim=True), outputs.max(dim=1, keepdim=True).values),
            dim=1,
        )
        spatial = self.original.spatial
        if self.spatial_mode == "regular":
            spatial_output = _conv2d_with_module(pooled, spatial.regular)
        elif self.spatial_mode == "gpu_deform":
            spatial_output = spatial(pooled)
        elif self.spatial_mode == "cpu_deform":
            spatial_output = _cpu_deformable_spatial(spatial, pooled)
        else:
            return outputs
        return outputs * torch.sigmoid(spatial_output)


def configure_variant(guide: nn.Module, variant: str) -> None:
    if variant in {"alignment_only", "full_original"}:
        return
    if variant == "attention_only":
        channel_mode, spatial_mode = "identity", "identity"
    elif variant == "adaptive_regular":
        channel_mode, spatial_mode = "adaptive", "regular"
    elif variant == "flatmax_regular":
        channel_mode, spatial_mode = "flatmax", "regular"
    elif variant == "flatmax_gpu_deform":
        channel_mode, spatial_mode = "flatmax", "gpu_deform"
    elif variant == "adaptive_cpu_deform":
        channel_mode, spatial_mode = "adaptive", "cpu_deform"
    elif variant == "full_deterministic_candidate":
        channel_mode, spatial_mode = "flatmax", "cpu_deform"
    else:
        raise ValueError(variant)
    for fusion in guide.fusion:
        fusion.cbam = DiagnosticCBAM(
            fusion.cbam,
            channel_mode=channel_mode,
            spatial_mode=spatial_mode,
        )


def apply_deterministic_candidate(guide: nn.Module) -> dict:
    configure_variant(guide, "full_deterministic_candidate")
    return deterministic_candidate_contract(guide)


def deterministic_candidate_contract(guide: nn.Module) -> dict:
    stages = []
    for index, fusion in enumerate(guide.fusion):
        cbam = fusion.cbam
        stages.append({
            "stage": index,
            "wrapper": type(cbam).__name__,
            "channel_mode": getattr(cbam, "channel_mode", None),
            "spatial_mode": getattr(cbam, "spatial_mode", None),
            "applied": (
                isinstance(cbam, DiagnosticCBAM)
                and cbam.channel_mode == "flatmax"
                and cbam.spatial_mode == "cpu_deform"
            ),
        })
    return {
        "candidate_id": CANDIDATE_ID,
        "applied": len(stages) == 3 and all(stage["applied"] for stage in stages),
        "channel_max": "flattened torch.max with first-index tie gradient",
        "deformable_spatial": "torchvision deform_conv2d on CPU",
        "cpu_threads": torch.get_num_threads(),
        "stages": stages,
    }
