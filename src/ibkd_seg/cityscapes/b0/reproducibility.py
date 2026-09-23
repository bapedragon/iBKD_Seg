"""Deterministic smoke execution and per-pixel CE without CUDA NLL2d reduction."""
import os

import torch
from torch.nn import functional as F


def configure_runtime():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Query chunking bounds the memory of the deterministic math implementation.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32": False, "sdpa_backend": "math", "cpu_threads": torch.get_num_threads(),
        "cross_entropy": "flatten_valid_pixels_2d_then_mean",
    }


def pixel_cross_entropy(logits, labels):
    """Same mean over valid pixels as CE(N,C,H,W, ignore_index=-1)."""
    resized = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=True)
    valid = labels != -1
    if not valid.any():
        raise ValueError("No valid labels")
    return F.cross_entropy(resized.permute(0,2,3,1)[valid], labels[valid])
