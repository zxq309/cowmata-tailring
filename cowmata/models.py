"""Multi-stage temporal model for untrimmed tail-ring IMU streams.

What changed and why
--------------------
The 20260818 model was a single-stage dilated TCN that classified each 2 Hz
point *independently* from its own 40.96 s window.  Two consequences followed
directly from that design, and both showed up in the results:

* every 50 Hz sample was re-encoded about 82 times per epoch, because
  neighbouring label points share almost all of their window;
* nothing in the model or the loss knew that a urination is one contiguous
  object, so the predicted point series fragmented and the event-level
  precision collapsed even where recall was near 1.

This module implements the standard remedy from the temporal
action-segmentation literature, which keeps the dilated-TCN backbone the
project already relies on and changes how it is used:

1. **A strided frame stem** turns the 50 Hz stream into one feature vector per
   0.5 s decision step.  All later stages run at 2 Hz, so a whole one-hour
   session is 7,200 steps and fits in memory at once.  Each raw sample is
   encoded exactly once per epoch instead of ~82 times.
2. **MS-TCN++ prediction generation** (Li et al., TPAMI 2020) with dual dilated
   layers: each layer mixes a small and a large dilation, so local shape and
   long context are seen at every depth instead of only at the top.  With the
   default 8 layers per stage the deepest dilation is 128 decision steps, i.e.
   64 s of one-sided context - comfortably longer than the longest annotated
   behaviour (a 30 s defecation) and short enough that the model is not asked
   to reason across an hour.
3. **Refinement stages** re-process the previous stage's probabilities.  This
   is what actually removes fragmentation, and it is supervised at every stage.
4. **A boundary head** in the spirit of ASRF (Ishikawa et al., WACV 2021).
   Regressing where events start and stop, and then using those boundaries to
   assemble intervals, attacks over-segmentation directly rather than hoping a
   smoothing loss will.

Causality
---------
``mode="causal"`` uses left-only padding everywhere, so no output depends on a
future sample - this is the online alerting model.  ``mode="offline"`` uses
symmetric padding and is for retrospective labelling, where the samples *after*
an event carry much of the discriminative information (how quickly the tail
drops separates urination from defecation).  Never mix the two between training
and inference; :func:`build_model` records the mode and the trainer writes it
into the checkpoint.

This module deliberately has no ``pytorch-tcn`` dependency.  The previous code
needed that package's streaming buffers because it ran the encoder on
overlapping windows; dense inference over a whole segment does not, and the
online path is served by :meth:`MultiTaskMSTCN.receptive_field_frames` plus
chunking with overlap, which is exact rather than approximately equivalent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .labels import EVENT_CODES

#: 50 Hz in, 2 Hz out.
DEFAULT_STEM_STRIDE = 25


def _pad_input(x: torch.Tensor, padding: int, causal: bool) -> torch.Tensor:
    if padding <= 0:
        return x
    return F.pad(x, (2 * padding, 0) if causal else (padding, padding))


def _factorise(value: int) -> tuple[int, ...]:
    """Small factors of the stem stride, largest last (25 -> (5, 5))."""

    remaining = int(value)
    factors: list[int] = []
    for candidate in (2, 3, 5, 7):
        while remaining % candidate == 0 and remaining > 1:
            factors.append(candidate)
            remaining //= candidate
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors) or (1,)


class DilatedResidualLayer(nn.Module):
    """One dilated residual layer, causal or symmetric."""

    def __init__(self, channels: int, dilation: int, *, causal: bool, dropout: float = 0.1) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.padding = self.dilation  # kernel size is fixed at 3
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=self.dilation)
        self.fuse = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = F.relu(self.conv(_pad_input(x, self.padding, self.causal)))
        out = self.dropout(self.fuse(out))
        out = x + out
        return out if mask is None else out * mask


class DualDilatedLayer(nn.Module):
    """MS-TCN++ prediction-generation layer.

    Mixes a dilation that grows with depth with one that shrinks, so a single
    layer sees both a short and a long neighbourhood.  In the original
    single-dilation stack the early layers can only ever see a couple of
    samples, which for a 0.5 s step means the first layers are blind to
    anything the length of a real behaviour.
    """

    def __init__(
        self,
        channels: int,
        dilation_growing: int,
        dilation_shrinking: int,
        *,
        causal: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.dilation_growing = int(dilation_growing)
        self.dilation_shrinking = int(dilation_shrinking)
        self.conv_growing = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=self.dilation_growing
        )
        self.conv_shrinking = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=self.dilation_shrinking
        )
        self.fuse = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.project = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        left = self.conv_growing(_pad_input(x, self.dilation_growing, self.causal))
        right = self.conv_shrinking(_pad_input(x, self.dilation_shrinking, self.causal))
        out = F.relu(self.fuse(torch.cat((left, right), dim=1)))
        out = self.dropout(self.project(out))
        out = x + out
        return out if mask is None else out * mask


class FrameStem(nn.Module):
    """50 Hz raw signal -> one feature vector per decision step.

    Implemented as stacked strided convolutions rather than a single stride-25
    convolution so the stem has a non-trivial receptive field (about 1.5 s) and
    can represent the shape of a movement, not just its average.  Set
    ``stride=1`` to feed an already-decimated feature sequence, for example the
    hand-crafted 120-feature bank or an external pretrained embedding, which is
    the cheapest way to test whether the temporal model or the representation is
    the bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = DEFAULT_STEM_STRIDE,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.stride = int(stride)
        self.causal = bool(causal)
        if self.stride == 1:
            self.layers = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size=1)])
            self.strides: tuple[int, ...] = (1,)
            self.kernels: tuple[int, ...] = (1,)
            return
        factors = _factorise(self.stride)
        self.strides = factors
        self.kernels = tuple(max(3, 2 * f + 1) for f in factors)
        layers: list[nn.Module] = []
        channels = in_channels
        for index, (factor, kernel) in enumerate(zip(factors, self.kernels)):
            width = out_channels if index == len(factors) - 1 else max(out_channels // 2, 16)
            layers.append(nn.Conv1d(channels, width, kernel_size=kernel, stride=factor))
            channels = width
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            kernel = self.kernels[index]
            stride = self.strides[index]
            padding = kernel - stride
            if padding > 0:
                x = F.pad(
                    x, (padding, 0) if self.causal else (padding // 2, padding - padding // 2)
                )
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x

    @property
    def receptive_field(self) -> int:
        field = 1
        cumulative = 1
        for kernel, stride in zip(self.kernels, self.strides):
            field += (kernel - 1) * cumulative
            cumulative *= stride
        return int(field)


class TaskHeads(nn.Module):
    """Posture, locomotion, per-event and boundary logits from shared features."""

    def __init__(self, channels: int, n_events: int) -> None:
        super().__init__()
        self.posture = nn.Conv1d(channels, 2, kernel_size=1)
        self.locomotion = nn.Conv1d(channels, 1, kernel_size=1)
        self.events = nn.Conv1d(channels, int(n_events), kernel_size=1)
        self.boundary = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "posture_logits": self.posture(features),
            "locomotion_logits": self.locomotion(features),
            "event_logits": self.events(features),
            "boundary_logits": self.boundary(features),
        }


class PredictionGeneration(nn.Module):
    """Stage 0 of MS-TCN++."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        n_events: int,
        *,
        layers: int = 8,
        causal: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.entry = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DualDilatedLayer(
                    channels,
                    dilation_growing=2**index,
                    dilation_shrinking=2 ** (layers - 1 - index),
                    causal=causal,
                    dropout=dropout,
                )
                for index in range(layers)
            ]
        )
        self.heads = TaskHeads(channels, n_events)
        self.n_layers = int(layers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = self.entry(x)
        if mask is not None:
            features = features * mask
        for layer in self.layers:
            features = layer(features, mask)
        return self.heads(features), features

    @property
    def receptive_field(self) -> int:
        return 1 + sum(2 * max(2**i, 2 ** (self.n_layers - 1 - i)) for i in range(self.n_layers))


class RefinementStage(nn.Module):
    """One refinement stage: re-reads the previous stage's own predictions."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        n_events: int,
        *,
        layers: int = 8,
        causal: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.entry = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(channels, 2**index, causal=causal, dropout=dropout)
                for index in range(layers)
            ]
        )
        self.heads = TaskHeads(channels, n_events)
        self.n_layers = int(layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.entry(x)
        if mask is not None:
            features = features * mask
        for layer in self.layers:
            features = layer(features, mask)
        return self.heads(features)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(2 * 2**index for index in range(self.n_layers))


class MultiTaskMSTCN(nn.Module):
    """The COWMATA model: strided stem, MS-TCN++ generation, refinement stages.

    ``forward`` returns *lists* of per-stage logits.  Every stage is supervised;
    that is the mechanism by which a refinement stage learns to clean up rather
    than to re-derive.  At inference only the last stage is used, which
    :meth:`predict` does for you.
    """

    def __init__(
        self,
        in_channels: int = 9,
        *,
        event_codes: Iterable[str] = EVENT_CODES,
        channels: int = 64,
        stage_layers: int = 8,
        refinement_stages: int = 3,
        stem_stride: int = DEFAULT_STEM_STRIDE,
        dropout: float = 0.1,
        causal: bool = False,
        sample_rate_hz: int = 50,
    ) -> None:
        super().__init__()
        self.event_codes = tuple(str(code) for code in event_codes)
        unknown = sorted(set(self.event_codes) - set(EVENT_CODES))
        if unknown:
            raise ValueError(f"unknown event codes: {unknown}")
        if refinement_stages < 0:
            raise ValueError("refinement_stages must be non-negative")
        self.causal = bool(causal)
        self.sample_rate_hz = int(sample_rate_hz)
        self.stem_stride = int(stem_stride)
        self.in_channels = int(in_channels)
        n_events = len(self.event_codes)

        self.stem = FrameStem(in_channels, channels, stride=stem_stride, causal=causal)
        self.generation = PredictionGeneration(
            channels, channels, n_events, layers=stage_layers, causal=causal, dropout=dropout
        )
        # A refinement stage reads the previous stage's task probabilities:
        # posture (2) + locomotion (1) + events (E) + boundary (1).
        refine_in = 2 + 1 + n_events + 1
        self.refinements = nn.ModuleList(
            [
                RefinementStage(
                    refine_in,
                    channels,
                    n_events,
                    layers=stage_layers,
                    causal=causal,
                    dropout=dropout,
                )
                for _ in range(refinement_stages)
            ]
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stage_probabilities(stage: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (
                torch.softmax(stage["posture_logits"], dim=1),
                torch.sigmoid(stage["locomotion_logits"]),
                torch.sigmoid(stage["event_logits"]),
                torch.sigmoid(stage["boundary_logits"]),
            ),
            dim=1,
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, list[torch.Tensor]]:
        """``inputs`` is ``(N, C, L)`` at the input rate; outputs are at 2 Hz.

        ``mask`` is ``(N, 1, L_out)`` and marks real (non-padded) decision steps
        in a batch of unequal-length segments.  Zeroing padded positions after
        every layer keeps them from leaking into their neighbours through the
        dilated convolutions.
        """

        features = self.stem(inputs)
        if mask is not None:
            if mask.shape[-1] != features.shape[-1]:
                raise ValueError(
                    f"mask length {mask.shape[-1]} does not match the decision-rate "
                    f"length {features.shape[-1]}"
                )
            features = features * mask
        stage, _ = self.generation(features, mask)
        stages = [stage]
        current = stage
        for refinement in self.refinements:
            current = refinement(self._stage_probabilities(current), mask)
            stages.append(current)
        return {
            "posture_logits": [s["posture_logits"] for s in stages],
            "locomotion_logits": [s["locomotion_logits"] for s in stages],
            "event_logits": [s["event_logits"] for s in stages],
            "boundary_logits": [s["boundary_logits"] for s in stages],
        }

    @torch.no_grad()
    def predict(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Final-stage probabilities, the only thing inference should consume."""

        out = self.forward(inputs, mask)
        return {
            "posture": torch.softmax(out["posture_logits"][-1], dim=1),
            "locomotion": torch.sigmoid(out["locomotion_logits"][-1]).squeeze(1),
            "events": torch.sigmoid(out["event_logits"][-1]),
            "boundary": torch.sigmoid(out["boundary_logits"][-1]).squeeze(1),
        }

    # ------------------------------------------------------------------
    @property
    def receptive_field_steps(self) -> int:
        """Receptive field in decision steps, counting every stage."""

        field = self.generation.receptive_field
        for refinement in self.refinements:
            field += refinement.receptive_field - 1
        return int(field)

    @property
    def receptive_field_frames(self) -> int:
        """Receptive field at the *input* rate.

        Online inference chunks the stream; the chunks must overlap by at least
        this many samples or the first outputs of each chunk are computed from
        truncated context and silently differ from the offline result.
        """

        return int(self.receptive_field_steps * self.stem_stride + self.stem.receptive_field)

    @property
    def receptive_field_seconds(self) -> float:
        return float(self.receptive_field_frames) / float(self.sample_rate_hz)

    def config(self) -> dict[str, object]:
        """Everything :func:`build_model` needs to rebuild this instance."""

        return {
            "in_channels": self.in_channels,
            "event_codes": list(self.event_codes),
            "channels": int(self.generation.entry.out_channels),
            "stage_layers": int(self.generation.n_layers),
            "refinement_stages": int(len(self.refinements)),
            "stem_stride": self.stem_stride,
            "sample_rate_hz": self.sample_rate_hz,
        }


def build_model(mode: str = "offline", **kwargs: object) -> MultiTaskMSTCN:
    """Factory shared by every training script.

    ``mode='causal'``  - online alerting; no output depends on a future sample.
    ``mode='offline'`` - retrospective labelling; symmetric context.
    """

    if mode not in {"causal", "offline"}:
        raise ValueError(f"unknown model mode: {mode!r}")
    kwargs.pop("causal", None)
    return MultiTaskMSTCN(causal=(mode == "causal"), **kwargs)  # type: ignore[arg-type]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ==========================================================================
# losses
# ==========================================================================
def smoothing_loss(
    logits: torch.Tensor, mask: torch.Tensor | None = None, *, clamp: float = 16.0
) -> torch.Tensor:
    """Truncated mean-squared error between adjacent log-probabilities (T-MSE).

    This is the MS-TCN over-segmentation penalty.  It costs nothing to compute
    and it is the difference between a probability series that flickers at every
    step and one that holds its value through an event.  Truncation keeps a
    genuine boundary - where the log-probability legitimately jumps - from
    dominating the gradient.
    """

    log_probability = F.log_softmax(logits, dim=1) if logits.shape[1] > 1 else F.logsigmoid(logits)
    difference = torch.clamp(
        (log_probability[..., 1:] - log_probability[..., :-1]).abs(), max=clamp
    )
    squared = difference**2
    if mask is not None:
        pair_mask = mask[..., 1:] * mask[..., :-1]
        return (squared * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    return squared.mean()


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return (raw * masks).sum() / masks.sum().clamp_min(1.0)


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross entropy over ``(N, C, L)`` logits with a ``(N, L)`` validity mask."""

    raw = F.cross_entropy(logits, targets.clamp_min(0), reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def multi_stage_loss(
    outputs: Mapping[str, Sequence[torch.Tensor]],
    targets: Mapping[str, torch.Tensor],
    *,
    event_pos_weight: torch.Tensor | None = None,
    locomotion_pos_weight: torch.Tensor | None = None,
    boundary_pos_weight: torch.Tensor | None = None,
    smoothing_weight: float = 0.15,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Sum the task losses over every stage.

    Supervising only the final stage would let the intermediate stages drift
    into any representation at all; MS-TCN supervises each one, and that is what
    makes the refinement behave as refinement.
    """

    posture_target = targets["posture_target"]
    posture_mask = targets["posture_mask"]
    locomotion_target = targets["locomotion_target"]
    locomotion_mask = targets["locomotion_mask"]
    event_target = targets["event_target"]
    event_mask = targets["event_mask"]
    boundary_target = targets["boundary_target"]
    boundary_mask = targets["boundary_mask"]

    total = torch.zeros((), device=event_target.device)
    parts: dict[str, torch.Tensor] = {}
    n_stages = len(outputs["event_logits"])
    for stage in range(n_stages):
        posture_logits = outputs["posture_logits"][stage]
        locomotion_logits = outputs["locomotion_logits"][stage].squeeze(1)
        event_logits = outputs["event_logits"][stage]
        boundary_logits = outputs["boundary_logits"][stage].squeeze(1)

        posture = masked_cross_entropy(posture_logits, posture_target, posture_mask)
        locomotion = masked_bce(
            locomotion_logits,
            locomotion_target,
            locomotion_mask,
            pos_weight=locomotion_pos_weight,
        )
        events = masked_bce(event_logits, event_target, event_mask, pos_weight=event_pos_weight)
        boundary = masked_bce(
            boundary_logits, boundary_target, boundary_mask, pos_weight=boundary_pos_weight
        )
        smooth = smoothing_loss(posture_logits, posture_mask.unsqueeze(1)) + smoothing_loss(
            event_logits, event_mask
        )
        stage_loss = (
            posture + locomotion + events + boundary_weight * boundary + smoothing_weight * smooth
        )
        total = total + stage_loss
        parts[f"stage{stage}"] = stage_loss.detach()
    parts["total"] = total
    return parts
