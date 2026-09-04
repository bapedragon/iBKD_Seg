"""Phase 1 teacher and spatial-guidance modules.

The iBKD implementation follows the frozen AAAI submission's Ours V1 source.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


TEACHER_CHANNELS = (16, 32, 64)
STUDENT_CHANNELS = 192
STUDENT_BLOCKS = 12
LG_STUDENT_BLOCKS = (0, 6, 11)


class BasicTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.a = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.a_af = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.a(inputs)
        outputs = self.a_bn(outputs)
        outputs = self.a_af(outputs)
        outputs = self.b(outputs)
        return self.b_bn(outputs)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.projection: nn.Module | None = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.projection_bn: nn.Module | None = nn.BatchNorm2d(
                out_channels, eps=1e-5, momentum=0.1
            )
        else:
            self.projection = None
            self.projection_bn = None
        self.transform = BasicTransform(in_channels, out_channels, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        if self.projection is not None:
            assert self.projection_bn is not None
            residual = self.projection_bn(self.projection(inputs))
        return self.activation(residual + self.transform(inputs))


class ResStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        blocks = [ResBlock(in_channels, out_channels, stride)]
        blocks.extend(ResBlock(out_channels, out_channels, 1) for _ in range(8))
        super().__init__(*blocks)


class ResNet56(nn.Module):
    """Official-LG-compatible CIFAR-style ResNet56 with a Pet classifier."""

    def __init__(self, num_classes: int = 37) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResStage(16, 16, stride=1)
        self.stage2 = ResStage(16, 32, stride=2)
        self.stage3 = ResStage(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(64, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.stem(inputs)
        feature1 = self.stage1(outputs)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        return feature1, feature2, feature3

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature = self.forward_features(inputs)[-1]
        return self.head(torch.flatten(self.pool(feature), 1))


def create_student(*, num_classes: int = 37, drop_path_rate: float = 0.1) -> nn.Module:
    import timm

    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=0.0,
    )


def forward_student_spatial(
    student: nn.Module, images: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    final_tokens, features = student.forward_intermediates(
        images,
        indices=tuple(range(STUDENT_BLOCKS)),
        norm=False,
        output_fmt="NCHW",
        intermediates_only=False,
    )
    if len(features) != STUDENT_BLOCKS:
        raise RuntimeError(f"Expected 12 DeiT block features, got {len(features)}")
    return list(features), student.forward_head(final_tokens)


class LocalityGuidance(nn.Module):
    """Official LG projections and summed stage-mean MSE."""

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(STUDENT_CHANNELS, channels, kernel_size=1)
            for channels in TEACHER_CHANNELS
        )

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(student_features) != STUDENT_BLOCKS or len(teacher_features) != 3:
            raise ValueError("LG expects 12 student blocks and three teacher stages")
        loss = teacher_features[0].new_zeros(())
        for block, projection, teacher_feature in zip(
            LG_STUDENT_BLOCKS, self.projections, teacher_features, strict=True
        ):
            student_feature = projection(student_features[block])
            target_size = (
                max(student_feature.shape[-2], teacher_feature.shape[-2]),
                max(student_feature.shape[-1], teacher_feature.shape[-1]),
            )
            if student_feature.shape[-2:] != target_size:
                student_feature = F.interpolate(
                    student_feature, size=target_size, mode="bilinear", align_corners=False
                )
            if teacher_feature.shape[-2:] != target_size:
                teacher_feature = F.interpolate(
                    teacher_feature, size=target_size, mode="bilinear", align_corners=False
                )
            loss = loss + F.mse_loss(student_feature.float(), teacher_feature.float())
        return loss


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        padding: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        points = kernel_size * kernel_size
        self.padding = (padding, padding)
        self.offset = nn.Conv2d(
            in_channels, 2 * points, kernel_size, padding=padding, bias=True
        )
        self.modulator = nn.Conv2d(
            in_channels, points, kernel_size, padding=padding, bias=True
        )
        self.regular = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torchvision.ops.deform_conv2d(
            input=inputs,
            offset=self.offset(inputs),
            weight=self.regular.weight,
            bias=self.regular.bias,
            padding=self.padding,
            mask=2.0 * torch.sigmoid(self.modulator(inputs)),
        )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(1, channels // 16)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.maximum = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(self.average(inputs)) + self.mlp(self.maximum(inputs)))


class DeformableCBAM(nn.Module):
    def __init__(self, channels: int, spatial_kernel_size: int = 5) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = DeformableConv2d(
            2,
            1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs * self.channel(inputs)
        pooled = torch.cat(
            (outputs.mean(dim=1, keepdim=True), outputs.max(dim=1, keepdim=True).values),
            dim=1,
        )
        return outputs * torch.sigmoid(self.spatial(pooled))


class ConvCrossAttention(nn.Module):
    def __init__(self, channels: int, *, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.cbam = DeformableCBAM(channels, spatial_kernel_size=5)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(
        self, student_feature: torch.Tensor, teacher_feature: torch.Tensor
    ) -> torch.Tensor:
        if student_feature.shape != teacher_feature.shape:
            raise ValueError("iBKD aligned student and teacher shapes must match")
        batch, channels, height, width = student_feature.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.flatten(2).transpose(1, 2).reshape(
                batch, height * width, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        query = split_heads(self.query(self.cbam(student_feature)))
        key = split_heads(self.key(teacher_feature))
        value = split_heads(self.value(teacher_feature))
        attention = torch.softmax((query @ key.transpose(-2, -1)) * self.scale, dim=-1)
        outputs = (attention @ value).transpose(1, 2).reshape(
            batch, height * width, channels
        )
        outputs = outputs.transpose(1, 2).reshape(batch, channels, height, width)
        return self.output(outputs)


class TransformerAggregationPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(3, STUDENT_BLOCKS))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != STUDENT_BLOCKS:
            raise ValueError(f"Expected 12 student features, got {len(features)}")
        stacked = torch.stack(tuple(features), dim=1)
        return torch.einsum("gl,bldhw->bgdhw", torch.softmax(self.weights, dim=-1), stacked)


class IBKD(nn.Module):
    """Submitted Ours V1 alignment/fusion module with larger-grid resize."""

    def __init__(self) -> None:
        super().__init__()
        self.aggregation = TransformerAggregationPooling()
        self.projections = nn.ModuleList(
            nn.Conv2d(STUDENT_CHANNELS, channels, 1) for channels in TEACHER_CHANNELS
        )
        self.fusion = nn.ModuleList(
            ConvCrossAttention(channels, num_heads=4) for channels in TEACHER_CHANNELS
        )

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(teacher_features) != 3:
            raise ValueError("iBKD expects three teacher stages")
        aggregated = self.aggregation(student_features)
        alignment_loss = aggregated.new_zeros(())
        fusion_loss = aggregated.new_zeros(())
        for stage, (teacher_feature, projection, fusion) in enumerate(
            zip(teacher_features, self.projections, self.fusion, strict=True)
        ):
            aligned = projection(aggregated[:, stage])
            target_size = (
                max(aligned.shape[-2], teacher_feature.shape[-2]),
                max(aligned.shape[-1], teacher_feature.shape[-1]),
            )
            if aligned.shape[-2:] != target_size:
                aligned = F.interpolate(
                    aligned, size=target_size, mode="bilinear", align_corners=False
                )
            if teacher_feature.shape[-2:] != target_size:
                teacher_feature = F.interpolate(
                    teacher_feature, size=target_size, mode="bilinear", align_corners=False
                )
            fused = fusion(aligned, teacher_feature)
            alignment_loss = alignment_loss + F.mse_loss(
                aligned.float(), teacher_feature.float()
            )
            fusion_loss = fusion_loss + F.mse_loss(fused.float(), teacher_feature.float())
        return alignment_loss, fusion_loss


def teacher_view(images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(images, size=(32, 32), mode="bilinear", align_corners=False)
