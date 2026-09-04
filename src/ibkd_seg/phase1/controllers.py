"""LG, ALG, and iBKD guidance schedules used by Phase 1."""

from __future__ import annotations

from typing import Any


class GuidanceController:
    def __init__(
        self,
        *,
        kind: str,
        beta: float = 2.5,
        threshold: float = -0.02,
        window: int = 50,
        warmup_epochs: int = 0,
    ) -> None:
        if kind not in {"lg", "alg", "ibkd"}:
            raise ValueError(f"Unknown controller kind {kind!r}")
        self.kind = kind
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.window = int(window)
        self.warmup_epochs = int(warmup_epochs)
        self.active = True
        self.stop_epoch: int | None = None
        self.losses: list[float] = []
        self.derivatives: list[float | None] = []
        self.smoothed_derivatives: list[float | None] = []
        self.beta_history: list[float] = []

    def beta_for_epoch(self, epoch: int) -> float:
        if epoch != len(self.beta_history) + 1:
            raise ValueError("Guidance beta requests must be consecutive and one-based")
        value = self.beta if self.active else 0.0
        self.beta_history.append(value)
        return value

    def _delta(self, epoch: int) -> float | None:
        if epoch < 2 or len(self.losses) < epoch:
            return None
        if epoch <= self.window:
            previous_mean = sum(self.losses[: epoch - 1]) / (epoch - 1)
            return (self.losses[epoch - 1] - previous_mean) / epoch
        return (self.losses[epoch - 1] - self.losses[epoch - self.window - 1]) / self.window

    def _smoothed(self, epoch: int) -> float | None:
        if epoch < 2:
            return None
        if epoch <= self.window:
            values = [self._delta(index) for index in range(2, epoch + 1)]
            finite = [float(value) for value in values if value is not None]
            if not finite:
                return None
            # ALG Eq. 16 uses the explicit 1/e normalization.  Ours V1's
            # supplied controller instead averages the available derivatives.
            denominator = epoch if self.kind == "alg" else len(finite)
            return sum(finite) / denominator
        if epoch < 2 * self.window:
            first = epoch - self.window + 1
            values = [self._delta(index) for index in range(first, epoch + 1)]
            finite = [float(value) for value in values if value is not None]
            return sum(finite) / len(finite) if finite else None
        first = epoch - self.window + 1
        total = sum(
            self.losses[index - 1] - self.losses[index - self.window - 1]
            for index in range(first, epoch + 1)
        )
        return total / (self.window**2)

    def observe(self, epoch: int, guidance_loss: float, *, beta_used: float) -> None:
        if beta_used <= 0.0:
            return
        if epoch != len(self.losses) + 1:
            raise ValueError("Guidance observations must be consecutive and one-based")
        self.losses.append(float(guidance_loss))
        if self.kind == "lg":
            self.derivatives.append(None)
            self.smoothed_derivatives.append(None)
            return
        derivative = self._delta(epoch)
        self.derivatives.append(derivative)
        smoothed = None if epoch < self.warmup_epochs else self._smoothed(epoch)
        self.smoothed_derivatives.append(smoothed)
        if smoothed is None:
            return
        crossed = smoothed >= self.threshold if self.kind == "alg" else smoothed > self.threshold
        if crossed:
            self.active = False
            self.stop_epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "beta_on": self.beta,
            "threshold": self.threshold,
            "smoothing_window": self.window,
            "warmup_epochs": self.warmup_epochs,
            "stop_comparison": "greater_or_equal" if self.kind == "alg" else "strictly_greater",
            "active": self.active,
            "stop_epoch": self.stop_epoch,
            "loss_history": list(self.losses),
            "derivative_history": list(self.derivatives),
            "smoothed_derivative_history": list(self.smoothed_derivatives),
            "beta_history": list(self.beta_history),
        }
