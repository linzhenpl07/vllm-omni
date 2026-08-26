# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spectral Self-Anchoring for autoregressive video, after FreqForcing.

An autoregressive video session degrades as it runs: measured on LingBot World
at 480x832 the picture is visibly wrong by tick 30 and unrecognisable by tick
100, while every frame-to-frame metric stays flat because a collapsed video
still changes smoothly. FreqForcing (arXiv:2607.27110) characterises that
degradation in the frequency domain -- error accumulation shows up as energy
drift in the DC and low-frequency bands, while high frequencies destabilise --
and corrects it at inference time, with no retraining.

The correction has two halves. A second attention branch reads a small cache of
frozen early frames, which are still trustworthy because they were produced
while the session was inside its pretrained horizon. Its output is then blended
into the ordinary sliding-window output *in the frequency domain*, taking only
the low-frequency part: the anchor supplies the slow structure that drifts, and
the local branch keeps the detail and the motion that the anchor, being frozen,
cannot know about.

This module is the arithmetic of that blend and the policy for choosing what to
anchor on. Both are deliberately free of the model: they take tensors and
return tensors, so they can be tested without a checkpoint or a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "AnchorFramePolicy",
    "SpectralAnchorConfig",
    "gaussian_low_pass",
    "spectral_anchor_fuse",
]


@dataclass(frozen=True)
class SpectralAnchorConfig:
    """Hyperparameters, defaulted to the values FreqForcing reports.

    ``sigma_t`` and ``sigma_xy`` are widths in normalised frequency, where 0.5
    is Nyquist, so 0.125 keeps roughly the lowest quarter of each axis.
    ``weight`` is how much of the anchor's low-frequency content replaces the
    local branch's: 0 disables the correction, 1 takes the anchor's low band
    outright.
    """

    sigma_t: float = 0.125
    sigma_xy: float = 0.125
    weight: float = 0.6
    anchor_capacity: int = 6
    frames_per_anchor: int = 3
    denoising_steps: int = 2

    def __post_init__(self) -> None:
        if not self.sigma_t > 0 or not self.sigma_xy > 0:
            raise ValueError("Gaussian widths must be positive.")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must lie in [0, 1], got {self.weight}.")
        if self.anchor_capacity < 1 or self.frames_per_anchor < 1:
            raise ValueError("anchor_capacity and frames_per_anchor must be at least 1.")
        if self.denoising_steps < 0:
            raise ValueError("denoising_steps must not be negative.")


def gaussian_low_pass(
    frames: int,
    height: int,
    width: int,
    *,
    sigma_t: float,
    sigma_xy: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Separable Gaussian low-pass over a ``frames x height x width`` grid.

    ``H(k_t, k_xy) = exp(-k_t^2 / 2 sigma_t^2 - ||k_xy||^2 / 2 sigma_xy^2)``,
    with frequencies in cycles per sample so the widths do not have to be
    rescaled when the grid does. Returned real and unnormalised: it multiplies a
    spectrum, it is not a distribution.
    """
    if min(frames, height, width) < 1:
        raise ValueError(f"grid must be positive in every axis, got {(frames, height, width)}.")
    k_t = torch.fft.fftfreq(frames, device=device, dtype=dtype).view(-1, 1, 1)
    k_y = torch.fft.fftfreq(height, device=device, dtype=dtype).view(1, -1, 1)
    k_x = torch.fft.fftfreq(width, device=device, dtype=dtype).view(1, 1, -1)
    exponent = -(k_t * k_t) / (2.0 * sigma_t * sigma_t) - (k_y * k_y + k_x * k_x) / (2.0 * sigma_xy * sigma_xy)
    return torch.exp(exponent)


def spectral_anchor_fuse(
    local: torch.Tensor,
    anchor: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    config: SpectralAnchorConfig | None = None,
) -> torch.Tensor:
    """Blend the anchor branch's low frequencies into the local branch.

    ``A = A_loc + w * H_lp * (A_anc - A_loc)``, evaluated on the 3D spectrum of
    the token grid and transformed back.

    Both inputs are attention outputs for the *current* chunk, laid out as
    ``[batch, tokens, heads, head_dim]`` with ``tokens == frames*height*width``
    in that nesting order -- the layout the model already produces, so no
    permutation is needed on the way in.

    The transform runs in float32 whatever the inputs are. bfloat16 has eight
    mantissa bits, and a difference of two coefficients followed by a round trip
    through an FFT is exactly the arithmetic that has none to spare; the result
    is cast back to the input dtype at the end.
    """
    config = config or SpectralAnchorConfig()
    if local.shape != anchor.shape:
        raise ValueError(f"branches must agree in shape, got {tuple(local.shape)} and {tuple(anchor.shape)}.")
    if local.ndim != 4:
        raise ValueError(f"expected [batch, tokens, heads, head_dim], got {tuple(local.shape)}.")
    tokens = local.shape[1]
    if tokens != frames * height * width:
        raise ValueError(f"tokens {tokens} does not factor as {frames}x{height}x{width}.")
    if config.weight == 0.0:
        return local

    batch, _, heads, head_dim = local.shape
    grid = (batch, frames, height, width, heads, head_dim)
    local_grid = local.reshape(grid).to(torch.float32)
    anchor_grid = anchor.reshape(grid).to(torch.float32)

    axes = (1, 2, 3)
    difference = torch.fft.fftn(anchor_grid - local_grid, dim=axes)
    envelope = gaussian_low_pass(
        frames,
        height,
        width,
        sigma_t=config.sigma_t,
        sigma_xy=config.sigma_xy,
        device=local.device,
    ).view(1, frames, height, width, 1, 1)
    # Linear in the spectrum, so filtering the difference and adding it back is
    # the same as filtering both branches -- one transform instead of two.
    correction = torch.fft.ifftn(difference * envelope, dim=axes).real
    fused = local_grid + config.weight * correction
    return fused.reshape(local.shape).to(local.dtype)


@dataclass
class AnchorFramePolicy:
    """Which latent frames to anchor on, and when to stop collecting.

    The anchor's value is that it is *early*: frames produced while the session
    was still inside its pretrained horizon have not yet accumulated the drift
    the anchor exists to correct. So the cache fills once, from the opening of
    the session, and is then frozen -- refreshing it from recent frames would
    anchor the session to its own drift and correct nothing.

    Collection is spread out rather than taken as one contiguous run: one frame
    in every ``frames_per_anchor`` covers more of the opening scene for the same
    number of cached frames.
    """

    config: SpectralAnchorConfig = field(default_factory=SpectralAnchorConfig)
    frames_seen: int = 0
    collected: list[int] = field(default_factory=list)

    @property
    def frozen(self) -> bool:
        return len(self.collected) >= self.config.anchor_capacity

    def offer(self, latent_frame_index: int) -> bool:
        """Record one generated latent frame; report whether it is anchored.

        Called once per generated latent frame, in order. Frames offered after
        the cache is full are counted and refused.
        """
        if latent_frame_index != self.frames_seen:
            raise ValueError(f"frames must be offered in order: expected {self.frames_seen}, got {latent_frame_index}.")
        self.frames_seen += 1
        if self.frozen:
            return False
        if latent_frame_index % self.config.frames_per_anchor:
            return False
        self.collected.append(latent_frame_index)
        return True

    def active(self, pretrained_latent_frames: int) -> bool:
        """Whether the correction should be applied yet.

        Inside the pretrained horizon the local branch is trustworthy on its own
        and the anchor has nothing to add, so the correction stays off until the
        session has run past the length the model was trained for -- which is
        also the point after which the cache is guaranteed to have filled.
        """
        return self.frozen and self.frames_seen > pretrained_latent_frames
