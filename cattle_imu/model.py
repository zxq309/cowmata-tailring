"""Causal multi-scale TCN models for continuous cattle-tail IMU streams.

The temporal backbone and streaming buffers come from the MIT-licensed
``pytorch-tcn`` package instead of a local reimplementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from .annotations import EVENT_CODES

try:
    from pytorch_tcn import TCN, TemporalConv1d
except ImportError as exc:  # pragma: no cover - exercised by environment check
    raise ImportError(
        "pytorch-tcn is required. Run `python -m pip install -e .` in the "
        "project environment."
    ) from exc


DEFAULT_EVENT_CONTEXT_SECONDS: dict[str, float] = {
    "STANDING_UP": 8.0,
    "LYING_DOWN": 8.0,
    "URINATION": 20.0,
    "DEFECATION": 30.0,
    "TAIL_RAISED": 15.0,
    "TAIL_WAGGING": 3.0,
}


class CausalTCNEncoder(nn.Module):
    """Shared timestamp-level encoder; output shape is ``(N, width, L)``."""

    def __init__(
        self,
        in_channels: int,
        *,
        width: int = 64,
        dropout: float = 0.10,
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        dilation_list = [int(value) for value in dilations]
        if not dilation_list or min(dilation_list) <= 0:
            raise ValueError("dilations must contain positive integers")
        self.width = int(width)
        self.network = TCN(
            num_inputs=int(in_channels),
            num_channels=[self.width] * len(dilation_list),
            kernel_size=5,
            dilations=dilation_list,
            dropout=float(dropout),
            causal=True,
            use_norm="weight_norm",
            activation="relu",
            kernel_initializer="xavier_uniform",
            use_skip_connections=True,
            input_shape="NCL",
        )

    def forward(self, inputs: torch.Tensor, *, inference: bool = False) -> torch.Tensor:
        return self.network(inputs, inference=inference)

    def reset_buffers(self) -> None:
        self.network.reset_buffers()


class CausalScaleHead(nn.Module):
    """Depthwise causal context filter followed by a pointwise output layer."""

    def __init__(self, channels: int, outputs: int, context_samples: int, dropout: float) -> None:
        super().__init__()
        if context_samples <= 0:
            raise ValueError("context_samples must be positive")
        self.context_samples = int(context_samples)
        self.temporal = TemporalConv1d(
            channels,
            channels,
            kernel_size=self.context_samples,
            groups=channels,
            bias=False,
            causal=True,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(channels, outputs, kernel_size=1)

    def forward(self, features: torch.Tensor, *, inference: bool = False) -> torch.Tensor:
        value = self.temporal(features, inference=inference)
        return self.output(self.dropout(self.activation(value)))

    def forward_last(self, features: torch.Tensor) -> torch.Tensor:
        """Compute only the final timestep.

        The causal conv's last output depends solely on the last
        ``context_samples`` feature positions, so a single unpadded ``F.conv1d``
        over that tail is equivalent to the full-length ``TemporalConv1d``'s
        last position while costing O(context_samples) instead of O(L * K).
        The huge kernels here (up to 1500 samples) otherwise dominate the batch.
        """
        if features.shape[-1] >= self.context_samples:
            tail = features[..., -self.context_samples :]
            value = nn.functional.conv1d(
                tail,
                self.temporal.weight,
                self.temporal.bias,
                stride=self.temporal.stride,
                padding=0,
                dilation=self.temporal.dilation,
                groups=self.temporal.groups,
            )
        else:
            # Short input (e.g. the environment smoke test) is shorter than the
            # kernel; fall back to the padded causal conv, which handles any
            # length via left-padding.
            value = self.temporal(features)[..., -1:]
        return self.output(self.dropout(self.activation(value)))

    def reset_buffers(self) -> None:
        self.temporal.reset_buffer()


class CausalMultiTaskTCN(nn.Module):
    """Shared causal encoder with posture, walking and independent event heads."""

    def __init__(
        self,
        in_channels: int = 8,
        *,
        event_codes: Iterable[str] = EVENT_CODES,
        sample_rate_hz: int = 50,
        width: int = 64,
        dropout: float = 0.10,
        event_context_seconds: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.event_codes = tuple(str(code) for code in event_codes)
        unknown = sorted(set(self.event_codes) - set(EVENT_CODES))
        if unknown:
            raise ValueError(f"unknown event codes: {unknown}")
        self.sample_rate_hz = int(sample_rate_hz)
        self.encoder = CausalTCNEncoder(
            in_channels,
            width=width,
            dropout=dropout,
        )
        contexts = dict(DEFAULT_EVENT_CONTEXT_SECONDS)
        if event_context_seconds is not None:
            contexts.update({str(key): float(value) for key, value in event_context_seconds.items()})
        self.posture_head = CausalScaleHead(
            width, 2, round(20.0 * self.sample_rate_hz), dropout
        )
        self.locomotion_head = CausalScaleHead(
            width, 1, round(4.0 * self.sample_rate_hz), dropout
        )
        self.event_heads = nn.ModuleDict(
            {
                code: CausalScaleHead(
                    width,
                    1,
                    max(1, round(contexts[code] * self.sample_rate_hz)),
                    dropout,
                )
                for code in self.event_codes
            }
        )

    def forward_dense(
        self,
        inputs: torch.Tensor,
        *,
        inference: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs, inference=inference)
        event_logits = torch.cat(
            [self.event_heads[code](features, inference=inference) for code in self.event_codes],
            dim=1,
        )
        return {
            "posture_logits": self.posture_head(features, inference=inference),
            "locomotion_logits": self.locomotion_head(features, inference=inference),
            "event_logits": event_logits,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        return {
            "posture_logits": self.posture_head.forward_last(features)[..., 0],
            "locomotion_logits": self.locomotion_head.forward_last(features)[..., 0],
            "event_logits": torch.cat(
                [self.event_heads[code].forward_last(features) for code in self.event_codes],
                dim=1,
            )[..., 0],
        }

    def reset_stream(self) -> None:
        self.encoder.reset_buffers()
        self.posture_head.reset_buffers()
        self.locomotion_head.reset_buffers()
        for head in self.event_heads.values():
            head.reset_buffers()


class DilatedTCNEncoder(nn.Module):
    """Non-causal counterpart of :class:`CausalTCNEncoder`.

    Retrospective analysis on a server does not need causality, and for event
    detection the samples *after* the label carry most of the discriminative
    information (how fast the tail comes back down separates urination from
    defecation).  Use this encoder for offline candidate mining and for the
    cross-cow evaluation; keep the causal one for any future online alerting.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        width: int = 64,
        dropout: float = 0.10,
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        dilation_list = [int(value) for value in dilations]
        if not dilation_list or min(dilation_list) <= 0:
            raise ValueError("dilations must contain positive integers")
        self.width = int(width)
        self.network = TCN(
            num_inputs=int(in_channels),
            num_channels=[self.width] * len(dilation_list),
            kernel_size=5,
            dilations=dilation_list,
            dropout=float(dropout),
            causal=False,
            use_norm="weight_norm",
            activation="relu",
            kernel_initializer="xavier_uniform",
            use_skip_connections=True,
            input_shape="NCL",
        )

    def forward(self, inputs: torch.Tensor, *, inference: bool = False) -> torch.Tensor:
        return self.network(inputs)

    def reset_buffers(self) -> None:  # pragma: no cover - no streaming state
        return None


class CenteredScaleHead(nn.Module):
    """Symmetric depthwise context filter followed by a pointwise output layer."""

    def __init__(self, channels: int, outputs: int, context_samples: int, dropout: float) -> None:
        super().__init__()
        if context_samples <= 0:
            raise ValueError("context_samples must be positive")
        # A symmetric 'same' convolution needs an odd kernel.
        self.context_samples = int(context_samples) | 1
        self.padding = self.context_samples // 2
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=self.context_samples,
            groups=channels,
            bias=False,
            padding=self.padding,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(channels, outputs, kernel_size=1)

    def forward(self, features: torch.Tensor, *, inference: bool = False) -> torch.Tensor:
        return self.output(self.dropout(self.activation(self.temporal(features))))

    def forward_center(self, features: torch.Tensor) -> torch.Tensor:
        """Only the middle timestep, without paying for the full-length conv."""

        length = features.shape[-1]
        center = length // 2
        half = self.padding
        if center - half >= 0 and center + half + 1 <= length:
            patch = features[..., center - half : center + half + 1]
            value = nn.functional.conv1d(
                patch,
                self.temporal.weight,
                self.temporal.bias,
                groups=self.temporal.groups,
            )
        else:
            value = self.temporal(features)[..., center : center + 1]
        return self.output(self.dropout(self.activation(value)))

    def reset_buffers(self) -> None:  # pragma: no cover - no streaming state
        return None


class OfflineMultiTaskTCN(nn.Module):
    """Non-causal multi-task model whose label position is the window centre."""

    def __init__(
        self,
        in_channels: int = 8,
        *,
        event_codes: Iterable[str] = EVENT_CODES,
        sample_rate_hz: int = 50,
        width: int = 64,
        dropout: float = 0.10,
        event_context_seconds: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.event_codes = tuple(str(code) for code in event_codes)
        unknown = sorted(set(self.event_codes) - set(EVENT_CODES))
        if unknown:
            raise ValueError(f"unknown event codes: {unknown}")
        self.sample_rate_hz = int(sample_rate_hz)
        self.encoder = DilatedTCNEncoder(in_channels, width=width, dropout=dropout)
        contexts = dict(DEFAULT_EVENT_CONTEXT_SECONDS)
        if event_context_seconds is not None:
            contexts.update({str(key): float(value) for key, value in event_context_seconds.items()})
        self.posture_head = CenteredScaleHead(width, 2, round(20.0 * self.sample_rate_hz), dropout)
        self.locomotion_head = CenteredScaleHead(width, 1, round(4.0 * self.sample_rate_hz), dropout)
        self.event_heads = nn.ModuleDict(
            {
                code: CenteredScaleHead(
                    width, 1, max(1, round(contexts[code] * self.sample_rate_hz)), dropout
                )
                for code in self.event_codes
            }
        )

    def forward_dense(self, inputs: torch.Tensor, *, inference: bool = False) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        event_logits = torch.cat(
            [self.event_heads[code](features) for code in self.event_codes], dim=1
        )
        return {
            "posture_logits": self.posture_head(features),
            "locomotion_logits": self.locomotion_head(features),
            "event_logits": event_logits,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(inputs)
        return {
            "posture_logits": self.posture_head.forward_center(features)[..., 0],
            "locomotion_logits": self.locomotion_head.forward_center(features)[..., 0],
            "event_logits": torch.cat(
                [self.event_heads[code].forward_center(features) for code in self.event_codes],
                dim=1,
            )[..., 0],
        }

    def reset_stream(self) -> None:  # pragma: no cover - no streaming state
        return None


def build_model(mode: str = "causal", **kwargs: object) -> nn.Module:
    """Factory shared by every training script.

    ``mode='causal'``   - online-capable, label at the window end.
    ``mode='offline'``  - retrospective, label at the window centre.
    The training script must record the mode in ``manifest.json``; mixing the
    two between training and inference silently destroys accuracy.
    """

    if mode == "causal":
        return CausalMultiTaskTCN(**kwargs)  # type: ignore[arg-type]
    if mode == "offline":
        return OfflineMultiTaskTCN(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown model mode: {mode!r}")


class EventOnlyTCN(nn.Module):
    """Optional specialist using the same causal backbone and per-event scales."""

    def __init__(
        self,
        in_channels: int = 8,
        num_events: int | None = None,
        *,
        event_codes: Iterable[str] | None = None,
        sample_rate_hz: int = 50,
        width: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        codes = tuple(event_codes or EVENT_CODES[: int(num_events or len(EVENT_CODES))])
        self.model = CausalMultiTaskTCN(
            in_channels,
            event_codes=codes,
            sample_rate_hz=sample_rate_hz,
            width=width,
            dropout=dropout,
        )
        self.event_codes = codes

    @property
    def encoder(self) -> CausalTCNEncoder:
        return self.model.encoder

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)["event_logits"]


# Compatibility names retained for old experiment imports. Their behavior now
# follows the causal hierarchical-label contract.
MultiTaskResTCN = CausalMultiTaskTCN
EventOnlyResTCN = EventOnlyTCN


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
