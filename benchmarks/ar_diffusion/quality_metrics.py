# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare what a session looks like at different serving resolutions.

Lowering resolution is the only lever that moves this model's chunk latency
far enough to matter, so the question "how much worse does it look" decides
whether the lever may be pulled at all. Nothing in tree answers it.

**The trap this module exists to avoid.** The same seed at two resolutions
does not produce the same video at two qualities. The noise tensor has a
different shape, so the sampled trajectory diverges and the two clips show
different content. A paired metric between them therefore measures content
divergence *plus* quality loss, and reporting it as "quality loss" would
overstate the damage by an unknown amount.

So a paired number is never reported on its own. Every comparison is read
against a floor measured the same way: the reference resolution against
*itself* at a different seed, which is two different videos at identical
quality. If the cross-resolution distance is not clearly above that floor,
the measurement did not resolve anything and :func:`format_report` says so
instead of printing a number that looks like an answer.

Expect the paired metrics to under-resolve. Because the two clips show
different content, PSNR and SSIM compare pixels that were never meant to line
up, and they stay inside the floor through degradation a viewer would call
obvious. Sharpness, needing no counterpart frame, separates a blurred clip
from a merely different one far more reliably. Both are reported; neither
replaces looking at the frames.

Metrics are implemented against ``torch`` alone so this adds no dependency.
LPIPS is used when ``lpips`` or ``torchmetrics`` happens to be importable,
because it tracks perceived degradation better than PSNR or SSIM, but it is
never required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# A clip is (frames, 3, height, width), float32, [0, 1].
Clip = torch.Tensor


def _validate(clip: Clip, name: str) -> None:
    if clip.dim() != 4 or clip.shape[1] != 3:
        raise ValueError(f"{name} must be (frames, 3, height, width), got {tuple(clip.shape)}")
    if clip.shape[0] == 0:
        raise ValueError(f"{name} has no frames")


def resample_to(clip: Clip, height: int, width: int) -> Clip:
    """Scale a clip to a common size for paired comparison.

    Comparison happens at the *reference* size because that is the size the
    viewer's window is: serving at a lower resolution means the output is
    scaled up for display, and the upscaling blur is part of what the viewer
    sees. Comparing at the lower size instead would hide it.
    """
    _validate(clip, "clip")
    if clip.shape[-2:] == (height, width):
        return clip
    return F.interpolate(clip, size=(height, width), mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def psnr(reference: Clip, candidate: Clip) -> float:
    """Mean per-frame PSNR in dB, for clips already at the same size."""
    _validate(reference, "reference")
    _validate(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    mse = ((reference - candidate) ** 2).flatten(1).mean(dim=1)
    return float((10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean())


def _gaussian_window(size: int, sigma: float, channels: int, dtype, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] @ kernel[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim(reference: Clip, candidate: Clip, *, window_size: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM over frames, Wang et al. defaults, data range 1.0."""
    _validate(reference, "reference")
    _validate(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    if min(reference.shape[-2:]) < window_size:
        raise ValueError(f"frames smaller than the {window_size}px SSIM window: {tuple(reference.shape[-2:])}")

    channels = reference.shape[1]
    window = _gaussian_window(window_size, sigma, channels, reference.dtype, reference.device)

    def filt(x: torch.Tensor) -> torch.Tensor:
        # No padding: the border is dropped rather than invented. Reflect-pad
        # here disagreed with scikit-image by more than the metric's own scale
        # on low-SSIM pairs, because padded pixels correlate with their mirror.
        return F.conv2d(x, window, groups=channels)

    mu_r, mu_c = filt(reference), filt(candidate)
    mu_r2, mu_c2, mu_rc = mu_r * mu_r, mu_c * mu_c, mu_r * mu_c
    sigma_r2 = filt(reference * reference) - mu_r2
    sigma_c2 = filt(candidate * candidate) - mu_c2
    sigma_rc = filt(reference * candidate) - mu_rc

    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_rc + c1) * (2 * sigma_rc + c2)
    denominator = (mu_r2 + mu_c2 + c1) * (sigma_r2 + sigma_c2 + c2)
    return float((numerator / denominator).mean())


def sharpness(clip: Clip) -> float:
    """Variance of the Laplacian on luminance -- a no-reference detail proxy.

    Unlike the paired metrics this needs no counterpart, so it survives the
    content-divergence problem better: two clips of different content still
    have comparable detail if they are equally sharp. It is still only a
    proxy, and it rises with noise as well as with detail, so it is reported
    against the same across-seed floor.
    """
    _validate(clip, "clip")
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=clip.dtype, device=clip.device)
    luma = (clip * weights[None, :, None, None]).sum(dim=1, keepdim=True)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=clip.dtype,
        device=clip.device,
    )[None, None]
    edges = F.conv2d(F.pad(luma, (1, 1, 1, 1), mode="reflect"), kernel)
    return float(edges.flatten(1).var(dim=1).mean())


SKY_LOCKED_THRESHOLD = 0.40
"""Above this share of sky-like pixels, a clip has lost its scene.

Fixed from a labelled set of 36 clips before the held-out check, and left
alone afterwards. The check then found a seventh sky-lock nobody had looked
at, so the number is not fitted to the eye that labelled it.
"""


def sky_locked_fraction(clip: Clip, *, first_frame: int = 24, stride: int = 4) -> float:
    """Share of the frame that is bright, blue-dominant and smooth.

    :func:`sharpness` cannot see the failure that matters most here. When the
    camera drifts off the scene and ends up staring at the sky, the clip is
    ruined and its Laplacian variance goes *up*: a dark head against a flat
    wash has strong edges and little else to average them against. The worst
    clip in one sweep had the highest sharpness of its group.

    So ask the question directly. A clip that has lost its scene is mostly one
    smooth bright expanse; a working one carries trunks, a path and a figure as
    well. This counts the expanse. It is a fraction, which matters: the first
    attempt at this used bottom-third detail *over* whole-frame detail and
    scored the sky-locked clips highest of all, because the denominator
    collapses exactly when the frame goes flat.

    Early frames are skipped. Every resolution opens on a flat wash while the
    first chunk resolves, which is a property of the causal decoder and not of
    the clip's quality.

    Read it in one direction only. Above :data:`SKY_LOCKED_THRESHOLD` the clip
    has lost its scene -- 7 of 7 in the labelled set, against one false
    positive in thirty. Below it, the clip is *not sky-locked*, which is not
    the same as good: a camera that ends up inside a bush fills the frame just
    as completely and scores 0.27. Nothing here certifies a clip.

    A judge covering the whole family was tried and rejected. The unifying
    property looked like low diversity -- one material filling the frame --
    so it scored the entropy of a coarse RGB histogram. It separated the
    labelled set and then failed held-out: among the clips it flagged were
    good ones at the reference resolution, because a dim monochrome forest is
    low-diversity by intent. It was measuring the prompt, not the failure.
    """
    _validate(clip, "clip")
    frames = clip[first_frame::stride]
    if frames.shape[0] == 0:
        frames = clip[-1:]
    red, green, blue = frames[:, 0], frames[:, 1], frames[:, 2]
    # A plain channel mean, not the luma weights :func:`sharpness` uses. The
    # threshold below was validated against this definition; changing it would
    # invalidate the number without changing what it is trying to say.
    local_std = _local_std(frames.mean(dim=1))
    sky = (blue > red + 0.02) & (blue > green + 0.02) & (frames.amax(dim=1) > 0.35) & (local_std < 0.03)
    return float(sky.to(clip.dtype).mean())


def _local_std(luma: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Standard deviation over a square window, one value per pixel."""
    size = 2 * radius + 1
    padded = F.pad(luma[:, None], (radius,) * 4, mode="reflect")
    mean = F.avg_pool2d(padded, size, stride=1)
    mean_square = F.avg_pool2d(padded**2, size, stride=1)
    return torch.sqrt(torch.clamp(mean_square - mean**2, min=0.0))[:, 0]


def load_lpips():
    """Return an LPIPS callable, or None when no implementation is installed."""
    try:
        import lpips as _lpips

        net = _lpips.LPIPS(net="alex")
        net.eval()

        def score(reference: Clip, candidate: Clip) -> float:
            with torch.no_grad():
                return float(net(reference * 2 - 1, candidate * 2 - 1).mean())

        return score
    except ImportError:
        pass
    try:
        from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as _lp

        def score(reference: Clip, candidate: Clip) -> float:
            with torch.no_grad():
                return float(_lp(candidate, reference, normalize=True))

        return score
    except ImportError:
        return None


@dataclass(frozen=True)
class ResolutionSample:
    """One clip generated at one resolution with one seed."""

    label: str
    height: int
    width: int
    seed: int
    frames: Clip

    def __post_init__(self) -> None:
        _validate(self.frames, f"frames for {self.label}")


@dataclass(frozen=True)
class PairedDistance:
    """Distances between two clips, compared at the reference size."""

    psnr_db: float
    ssim: float
    lpips: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {"psnr_db": self.psnr_db, "ssim": self.ssim, "lpips": self.lpips}


def paired_distance(reference: ResolutionSample, candidate: ResolutionSample, *, lpips=None) -> PairedDistance:
    """Distance from candidate to reference, both at the reference's size."""
    target = resample_to(candidate.frames, reference.height, reference.width)
    frames = min(reference.frames.shape[0], target.shape[0])
    left, right = reference.frames[:frames], target[:frames]
    return PairedDistance(
        psnr_db=psnr(left, right),
        ssim=ssim(left, right),
        lpips=None if lpips is None else lpips(left, right),
    )


@dataclass(frozen=True)
class DivergenceFloor:
    """What two clips of *different content but equal quality* score.

    Built from several seeds at the reference resolution. Any cross-resolution
    distance must beat this to mean anything: it is the part of the distance
    that resolution cannot explain.
    """

    seeds: tuple[int, ...]
    distances: tuple[PairedDistance, ...]
    sharpness_values: tuple[float, ...]

    @property
    def resolved(self) -> bool:
        """Whether enough seeds were run for the floor to exist at all."""
        return len(self.distances) >= 1

    def worst(self, metric: str) -> float | None:
        """The floor value: the *most degraded* score content alone can cause."""
        values = [d.as_dict()[metric] for d in self.distances]
        present = [v for v in values if v is not None]
        if not present:
            return None
        # PSNR and SSIM fall as clips differ; LPIPS rises.
        return max(present) if metric == "lpips" else min(present)

    @property
    def sharpness_spread(self) -> float | None:
        if len(self.sharpness_values) < 2:
            return None
        return max(self.sharpness_values) - min(self.sharpness_values)


def measure_divergence_floor(samples: Sequence[ResolutionSample], *, lpips=None) -> DivergenceFloor:
    """Score same-resolution clips from different seeds against each other."""
    if len(samples) < 2:
        return DivergenceFloor(seeds=tuple(s.seed for s in samples), distances=(), sharpness_values=())
    sizes = {(s.height, s.width) for s in samples}
    if len(sizes) != 1:
        raise ValueError(f"the floor must be measured at one resolution, got {sorted(sizes)}")
    if len({s.seed for s in samples}) != len(samples):
        raise ValueError("the floor needs distinct seeds; identical seeds would measure nothing")
    head, rest = samples[0], samples[1:]
    return DivergenceFloor(
        seeds=tuple(s.seed for s in samples),
        distances=tuple(paired_distance(head, other, lpips=lpips) for other in rest),
        sharpness_values=tuple(sharpness(s.frames) for s in samples),
    )


@dataclass
class QualityReport:
    """Cross-resolution distances, plus the floor they have to be read against."""

    reference: ResolutionSample
    candidates: list[ResolutionSample] = field(default_factory=list)
    distances: list[PairedDistance] = field(default_factory=list)
    sharpness_values: list[float] = field(default_factory=list)
    sky_values: list[float] = field(default_factory=list)
    reference_sharpness: float = 0.0
    reference_sky: float = 0.0
    floor: DivergenceFloor | None = None

    def resolves(self, index: int, metric: str) -> bool | None:
        """Did this candidate beat the floor on this metric?

        ``None`` means unanswerable: no floor was measured, or the metric was
        unavailable. That is deliberately distinct from ``False``.
        """
        if self.floor is None or not self.floor.resolved:
            return None
        limit = self.floor.worst(metric)
        value = self.distances[index].as_dict()[metric]
        if limit is None or value is None:
            return None
        return value > limit if metric == "lpips" else value < limit

    def sharpness_resolves(self, index: int) -> bool | None:
        """Did this candidate's detail level move further than seeds alone do?

        Sharpness needs no counterpart frame, so content divergence perturbs it
        far less than it perturbs the paired metrics. In practice this is the
        measure that separates a blurred clip from a merely different one --
        the paired metrics can stay inside the floor through degradation a
        viewer would call obvious.
        """
        if self.floor is None:
            return None
        spread = self.floor.sharpness_spread
        if spread is None:
            return None
        return abs(self.sharpness_values[index] - self.reference_sharpness) > spread

    def lost_the_scene(self, index: int) -> bool:
        """Has this candidate's camera drifted off the scene into the sky?

        No floor is consulted. This is not a distance from the reference, it is
        a property of the clip on its own: either most of the frame is sky or it
        is not. A reference clip that scores above the line is itself broken.
        """
        return self.sky_values[index] > SKY_LOCKED_THRESHOLD


def build_report(
    reference: ResolutionSample,
    candidates: Sequence[ResolutionSample],
    *,
    floor_samples: Sequence[ResolutionSample] = (),
    lpips=None,
) -> QualityReport:
    """Score every candidate against the reference, with the floor attached."""
    report = QualityReport(
        reference=reference,
        reference_sharpness=sharpness(reference.frames),
        reference_sky=sky_locked_fraction(reference.frames),
    )
    for candidate in candidates:
        report.candidates.append(candidate)
        report.distances.append(paired_distance(reference, candidate, lpips=lpips))
        # At the reference size, not the candidate's own. Laplacian variance is
        # per-pixel, so a smaller frame scores *higher* for identical content --
        # the same detail spans fewer pixels. Measuring at native size would
        # therefore report lower resolutions as sharper, which is backwards.
        report.sharpness_values.append(sharpness(resample_to(candidate.frames, reference.height, reference.width)))
        # At native size. This is a share of the frame, so it does not change
        # with resampling, and resampling would only blur the smoothness test.
        report.sky_values.append(sky_locked_fraction(candidate.frames))
    if floor_samples:
        report.floor = measure_divergence_floor([reference, *floor_samples], lpips=lpips)
    return report


_VERDICTS = {
    True: "below floor -- resolution is visible",
    False: "WITHIN FLOOR -- resolves nothing",
    None: "unreadable without a floor",
}

_SHARPNESS_VERDICTS = {
    True: "moved further than seeds alone do",
    False: "WITHIN SEED SPREAD -- resolves nothing",
    None: "unreadable without a floor",
}


def format_report(report: QualityReport) -> str:
    """Render the report, refusing to present unresolved numbers as answers."""
    lines: list[str] = []
    ref = report.reference
    lines.append(f"reference  {ref.label}  {ref.width}x{ref.height}  seed={ref.seed}  frames={ref.frames.shape[0]}")
    lines.append(f"           sharpness {report.reference_sharpness:.4e}")
    lines.append(f"           sky       {report.reference_sky:.3f}")
    if report.reference_sky > SKY_LOCKED_THRESHOLD:
        lines.append("           REFERENCE HAS LOST ITS SCENE -- every distance below is measured")
        lines.append("           against a broken clip. Re-run the reference on another seed.")
    lines.append("")

    floor = report.floor
    if floor is None or not floor.resolved:
        lines.append("NO DIVERGENCE FLOOR MEASURED.")
        lines.append("")
        lines.append("  Distances below mix quality loss with content divergence: the same seed at")
        lines.append("  a different resolution produces a different video, not the same video at a")
        lines.append("  lower quality. Without a same-resolution/different-seed control there is no")
        lines.append("  way to tell the two apart, so these are NOT quality numbers.")
        lines.append("  Re-run with --floor-seeds to make them readable.")
    else:
        lines.append(f"divergence floor  seeds={list(floor.seeds)}  (different content, identical quality)")
        for metric in ("psnr_db", "ssim", "lpips"):
            value = floor.worst(metric)
            if value is not None:
                lines.append(f"  {metric:<8} {value:>10.5f}   distances must beat this to mean anything")
        spread = floor.sharpness_spread
        if spread is not None:
            lines.append(f"  {'sharp':<8} {spread:>10.4e}   spread across seeds at the reference size")
    lines.append("")

    for index, candidate in enumerate(report.candidates):
        distance = report.distances[index]
        lines.append(f"{candidate.label}  {candidate.width}x{candidate.height}  seed={candidate.seed}")
        for metric in ("psnr_db", "ssim", "lpips"):
            value = distance.as_dict()[metric]
            if value is None:
                lines.append(f"  {metric:<8}          -   not installed")
                continue
            note = _VERDICTS[report.resolves(index, metric)]
            lines.append(f"  {metric:<8} {value:>10.5f}   {note}")
        own = report.sharpness_values[index]
        ratio = own / report.reference_sharpness if report.reference_sharpness else float("nan")
        note = _SHARPNESS_VERDICTS[report.sharpness_resolves(index)]
        lines.append(f"  {'sharp':<8} {own:>10.4e}   {ratio:.3f}x reference -- {note}")
        sky = report.sky_values[index]
        if report.lost_the_scene(index):
            lines.append(f"  {'sky':<8} {sky:>10.3f}   LOST THE SCENE -- the camera is looking at the sky")
        else:
            lines.append(f"  {'sky':<8} {sky:>10.3f}   not sky-locked (which is not the same as good)")
        lines.append("")

    lines.append("Paired metrics are an upper bound on damage, never a lower bound: whatever part")
    lines.append("of the distance the floor explains is content, not quality.")
    lines.append("")
    lines.append("Expect PSNR and SSIM to stay inside the floor through degradation a viewer would")
    lines.append("call obvious -- they compare pixels that were never meant to line up. Sharpness")
    lines.append("is the more decisive signal here. Neither replaces looking at the frames.")
    lines.append("")
    lines.append("Sharpness cannot see a camera that has drifted off the scene -- it goes UP when a")
    lines.append("clip fills with sky. Read the sky line in one direction: over the line the clip is")
    lines.append("gone, under it only means not-sky-locked. A camera buried in foliage fills the")
    lines.append("frame just as completely and scores about 0.27, so nothing here certifies a clip.")
    return "\n".join(lines)


def frames_to_png_bytes(clip: Clip) -> list[bytes]:
    """Encode frames as PNG, so a human can look at what the numbers describe."""
    import io

    from PIL import Image

    _validate(clip, "clip")
    array = (clip.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    encoded: list[bytes] = []
    for frame in array:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        encoded.append(buffer.getvalue())
    return encoded


__all__ = [
    "DivergenceFloor",
    "PairedDistance",
    "QualityReport",
    "ResolutionSample",
    "build_report",
    "format_report",
    "frames_to_png_bytes",
    "load_lpips",
    "measure_divergence_floor",
    "paired_distance",
    "psnr",
    "resample_to",
    "sharpness",
    "ssim",
]
