# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the resolution-quality comparison.

Synthetic clips stand in for generated video: the metrics and, more
importantly, the reporting rules are what these verify. The rule that matters
is that a paired distance is never presented as a quality number unless a
same-resolution/different-seed floor was measured to read it against.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.ar_diffusion.quality_metrics import (
    DivergenceFloor,
    ResolutionSample,
    build_report,
    format_report,
    frames_to_png_bytes,
    measure_divergence_floor,
    paired_distance,
    psnr,
    resample_to,
    sharpness,
    ssim,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _clip(frames=3, height=64, width=64, *, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(frames, 3, height, width, generator=generator)


def _structured(seed, frames=3, height=64, width=64):
    """Smooth, structured content -- closer to video than white noise.

    White noise is the degenerate case: destroying it leaves something no more
    similar to the original than an unrelated clip is, so every metric reads
    zero and nothing can be distinguished. Real frames have low-frequency
    structure that survives blurring, which is what makes the comparison
    meaningful at all.
    """
    generator = torch.Generator().manual_seed(seed)
    field = torch.nn.functional.interpolate(
        torch.rand(frames, 3, 4, 4, generator=generator),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    ys = torch.linspace(0, 1, height)[None, None, :, None]
    xs = torch.linspace(0, 1, width)[None, None, None, :]
    bars = ((torch.sin(xs * 18 + seed) + torch.cos(ys * 14)) * 0.25 + 0.5).expand(frames, 3, height, width)
    return (field * 0.6 + bars * 0.4).clamp(0, 1)


def _sample(label, height, width, seed, *, frames=3, clip=None):
    return ResolutionSample(
        label=label,
        height=height,
        width=width,
        seed=seed,
        frames=_clip(frames, height, width, seed=seed) if clip is None else clip,
    )


# ── Metrics ────────────────────────────────────────────────────────────────


def test_identical_clips_score_perfectly():
    clip = _clip()
    assert psnr(clip, clip) > 100.0
    assert ssim(clip, clip) == pytest.approx(1.0, abs=1e-5)


def test_ssim_matches_an_independent_implementation():
    """Anchor the hand-written SSIM against scikit-image, not against itself.

    A metric verified only by its own properties can be self-consistently
    wrong; every number this module reports rests on this one being right.
    """
    skimage = pytest.importorskip("skimage.metrics")
    reference, candidate = _clip(1, seed=1), _clip(1, seed=2)

    mine = ssim(reference, candidate)
    theirs = skimage.structural_similarity(
        reference[0].permute(1, 2, 0).numpy(),
        candidate[0].permute(1, 2, 0).numpy(),
        channel_axis=2,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert mine == pytest.approx(theirs, abs=2e-3)


def test_psnr_matches_its_definition():
    reference = torch.zeros(1, 3, 32, 32)
    candidate = torch.full((1, 3, 32, 32), 0.1)
    assert psnr(reference, candidate) == pytest.approx(20.0, abs=1e-4)


def test_blurring_lowers_sharpness():
    clip = _clip(1, 64, 64)
    blurred = resample_to(resample_to(clip, 16, 16), 64, 64)
    assert sharpness(blurred) < sharpness(clip)


def test_resampling_compares_at_the_reference_size():
    """A downscaled clip is scored after being scaled back up, as displayed."""
    reference = _sample("ref", 64, 64, seed=1)
    small = ResolutionSample("small", 32, 32, 1, resample_to(reference.frames, 32, 32))

    distance = paired_distance(reference, small)

    assert distance.psnr_db < 100.0  # the round trip really did lose something
    assert distance.ssim < 1.0


def test_paired_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr(_clip(2), _clip(3))


# ── The divergence floor ───────────────────────────────────────────────────


def test_the_floor_needs_distinct_seeds():
    """Identical seeds would report a perfect floor and hide every result."""
    with pytest.raises(ValueError, match="distinct seeds"):
        measure_divergence_floor([_sample("a", 64, 64, 7), _sample("b", 64, 64, 7)])


def test_the_floor_must_be_measured_at_one_resolution():
    with pytest.raises(ValueError, match="one resolution"):
        measure_divergence_floor([_sample("a", 64, 64, 1), _sample("b", 32, 32, 2)])


def test_a_single_sample_yields_no_floor():
    floor = measure_divergence_floor([_sample("a", 64, 64, 1)])
    assert not floor.resolved
    assert floor.worst("ssim") is None


def test_the_floor_is_the_most_degraded_score_content_alone_causes():
    near = _sample("near", 64, 64, 2, clip=_clip(3, 64, 64, seed=1) * 0.99)
    far = _sample("far", 64, 64, 3)
    floor = measure_divergence_floor([_sample("ref", 64, 64, 1), near, far])

    # SSIM falls as clips differ, so the floor is the minimum...
    assert floor.worst("ssim") == min(d.ssim for d in floor.distances)
    # ...while LPIPS rises, so it would be the maximum.
    rising = DivergenceFloor(seeds=(1, 2), distances=floor.distances, sharpness_values=())
    assert rising.worst("psnr_db") == min(d.psnr_db for d in floor.distances)


# ── Reporting rules ────────────────────────────────────────────────────────


def test_a_report_without_a_floor_refuses_to_call_its_numbers_quality():
    reference = _sample("832x480", 48, 64, 1)
    report = build_report(reference, [_sample("384x288", 24, 32, 1)])

    assert report.resolves(0, "ssim") is None
    text = format_report(report)
    assert "NO DIVERGENCE FLOOR MEASURED" in text
    assert "NOT quality numbers" in text


def test_a_distance_inside_the_floor_is_reported_as_resolving_nothing():
    """Two unrelated clips differ as much as any resolution change would.

    This is the failure the floor exists to catch: without it, the same
    distance would read as a large quality loss.
    """
    reference = _sample("ref", 64, 64, 1)
    candidate = _sample("cand", 64, 64, 2)
    floor_sample = _sample("floor", 64, 64, 3)

    report = build_report(reference, [candidate], floor_samples=[floor_sample])

    assert report.resolves(0, "ssim") is False
    assert "WITHIN FLOOR" in format_report(report)


def test_severe_degradation_beats_the_floor():
    """Degrade far enough and the paired metrics do resolve it.

    The candidate has different content *and* is destroyed, which is the real
    shape of a cross-resolution comparison -- not the reference blurred.
    """
    reference = _sample("ref", 64, 64, 1, clip=_structured(1))
    destroyed = ResolutionSample("destroyed", 4, 4, 2, resample_to(_structured(2), 4, 4))
    floor_sample = _sample("floor", 64, 64, 3, clip=_structured(3))

    report = build_report(reference, [destroyed], floor_samples=[floor_sample])

    assert report.resolves(0, "ssim") is True
    assert "resolution is visible" in format_report(report)


def test_moderate_degradation_does_not_resolve_on_paired_metrics():
    """The finding that decides how this tool may be read.

    An 8x8 round trip is degradation nobody would miss, yet its SSIM against
    the reference stays *better* than the floor, because the reference and the
    candidate show different content and paired metrics compare pixels that
    were never meant to line up. Reporting that SSIM as a quality number would
    be wrong in both directions at once.
    """
    reference = _sample("ref", 64, 64, 1, clip=_structured(1))
    blurred = ResolutionSample("blurred", 8, 8, 2, resample_to(_structured(2), 8, 8))
    floor_sample = _sample("floor", 64, 64, 3, clip=_structured(3))

    report = build_report(reference, [blurred], floor_samples=[floor_sample])

    assert report.resolves(0, "ssim") is False
    # Sharpness, needing no counterpart frame, is not fooled.
    assert report.sharpness_resolves(0) is True


def test_sharpness_is_measured_at_the_reference_size():
    """Otherwise a lower resolution reports as *sharper*, which is backwards.

    Laplacian variance is per-pixel: the same content in a smaller frame
    occupies fewer pixels, so its detail sits at higher spatial frequency and
    scores higher. Comparing native sizes would rank a downscaled clip above
    the reference it was downscaled from.
    """
    base = _structured(1)
    reference = _sample("ref", 64, 64, 1, clip=base)
    small = ResolutionSample("small", 16, 16, 1, resample_to(base, 16, 16))

    # The confound is real: at its own size the small clip looks sharper.
    assert sharpness(small.frames) > sharpness(reference.frames)

    report = build_report(reference, [small])

    assert report.sharpness_values[0] < report.reference_sharpness


def test_sharpness_has_no_verdict_without_a_floor():
    report = build_report(_sample("ref", 64, 64, 1), [_sample("c", 32, 32, 1)])
    assert report.sharpness_resolves(0) is None


def test_the_report_warns_that_paired_metrics_under_resolve():
    report = build_report(_sample("ref", 64, 64, 1), [], floor_samples=[_sample("f", 64, 64, 2)])
    text = format_report(report)
    assert "stay inside the floor through degradation a viewer would" in text
    assert "looking at the frames" in text


def test_the_report_always_states_that_paired_numbers_are_an_upper_bound():
    report = build_report(_sample("ref", 64, 64, 1), [], floor_samples=[_sample("f", 64, 64, 2)])
    assert "upper bound on damage" in format_report(report)


def test_lpips_is_optional_and_reported_as_absent():
    report = build_report(_sample("ref", 64, 64, 1), [_sample("c", 32, 32, 1)])
    assert report.distances[0].lpips is None
    assert "not installed" in format_report(report)


def test_lpips_is_used_when_available():
    calls = []

    def fake_lpips(reference, candidate):
        calls.append((reference.shape, candidate.shape))
        return 0.42

    report = build_report(_sample("ref", 64, 64, 1), [_sample("c", 32, 32, 1)], lpips=fake_lpips)

    assert report.distances[0].lpips == 0.42
    # The candidate was scored at the reference size, not its own.
    assert calls[0][0] == calls[0][1]


# ── Frames on disk ─────────────────────────────────────────────────────────


def test_frames_encode_to_png_so_a_human_can_look():
    pytest.importorskip("PIL")
    encoded = frames_to_png_bytes(_clip(2, 16, 16))
    assert len(encoded) == 2
    assert all(blob.startswith(b"\x89PNG") for blob in encoded)


# ── CLI: everything that does not need a device ────────────────────────────


def _cli():
    from benchmarks.ar_diffusion import run_quality_vs_resolution

    return run_quality_vs_resolution


def test_resolution_alignment_is_rejected_before_a_model_loads():
    """A 30-second model load is a bad place to discover a bad resolution."""
    import argparse

    cli = _cli()
    assert cli.parse_resolution("832x480") == (832, 480)
    for bad in ("832", "832x0", "hello", "830x480"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_resolution(bad)


def test_generated_runs_round_trip_through_disk(tmp_path):
    """Frames written by generate must read back as the same clip."""
    pytest.importorskip("PIL")
    import json

    cli = _cli()
    clip = _structured(1, frames=2, height=32, width=48)
    directory = tmp_path / "48x32_seed0"
    directory.mkdir()
    for index, blob in enumerate(frames_to_png_bytes(clip)):
        (directory / f"frame_{index:05d}.png").write_bytes(blob)
    (directory / cli.MANIFEST).write_text(
        json.dumps({"label": "48x32_seed0", "width": 48, "height": 32, "seed": 0}), encoding="utf-8"
    )

    loaded = cli.load_sample(directory)

    assert (loaded.height, loaded.width, loaded.seed) == (32, 48, 0)
    # PNG is 8-bit, so the round trip is exact only to quantisation.
    assert torch.allclose(loaded.frames, clip, atol=1.0 / 255)


def test_loading_a_directory_that_is_not_a_run_says_so(tmp_path):
    cli = _cli()
    with pytest.raises(FileNotFoundError, match="not a generated run"):
        cli.load_sample(tmp_path)


def test_decoder_output_is_normalised_to_a_clip():
    cli = _cli()
    # [B, 3, T, H, W] in [-1, 1], as a Wan decoder emits.
    frames = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 5).reshape(1, 3, 2, 4, 5)
    clip = cli._frames_to_clip(frames)
    assert clip.shape == (2, 3, 4, 5)
    assert float(clip.min()) >= 0.0 and float(clip.max()) <= 1.0


def test_a_batched_decode_is_refused_rather_than_silently_taking_one():
    cli = _cli()
    with pytest.raises(ValueError, match="one session per decode"):
        cli._frames_to_clip(torch.zeros(2, 3, 2, 4, 5))


def _write_run(cli, directory, prompt, *, width=48, height=32, seed=0):
    """A minimal run on disk: two frames and a manifest naming its prompt."""
    import json

    directory.mkdir(parents=True, exist_ok=True)
    clip = _structured(1, frames=2, height=height, width=width)
    for index, blob in enumerate(frames_to_png_bytes(clip)):
        (directory / f"frame_{index:05d}.png").write_bytes(blob)
    (directory / cli.MANIFEST).write_text(
        json.dumps(
            {
                "label": directory.name,
                "width": width,
                "height": height,
                "seed": seed,
                "prompt": prompt,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_one_prompt_keeps_the_original_run_name():
    """Runs generated before prompts were repeatable stay addressable."""
    cli = _cli()
    assert cli.run_label(512, 320, 0) == "512x320_seed0"
    assert cli.run_label(512, 320, 1, 0, prompt_count=1) == "512x320_seed1"


def test_several_prompts_get_distinct_run_names():
    cli = _cli()
    labels = {cli.run_label(512, 320, seed, index, prompt_count=3) for index in range(3) for seed in (0, 1)}
    assert len(labels) == 6
    assert "512x320_p2_seed1" in labels


def test_extra_seeds_reach_the_candidate_only_when_asked(monkeypatch):
    """The default keeps the floor's meaning; --all-seeds repeats the candidate.

    Without this flag a candidate resolution is judged on exactly one sample,
    which is not enough to call a resolution unusable.
    """
    import argparse
    import asyncio

    cli = _cli()
    seen = []

    async def fake_generate_one(args, *, width, height, seed, prompt_index=0, prompt_count=1):
        label = cli.run_label(width, height, seed, prompt_index, prompt_count=prompt_count)
        seen.append(label)
        return args.out / label

    monkeypatch.setattr(cli, "generate_one", fake_generate_one)

    def run(all_seeds):
        seen.clear()
        args = argparse.Namespace(
            out=Path("/nowhere"),
            resolution=[(768, 480), (512, 320)],
            prompt=["a", "b"],
            seed=0,
            floor_seeds=1,
            all_seeds=all_seeds,
        )
        asyncio.run(cli._generate(args))
        return list(seen)

    default = run(False)
    # Reference gets both seeds, candidate only one -- per prompt.
    assert default == [
        "768x480_p0_seed0",
        "768x480_p0_seed1",
        "512x320_p0_seed0",
        "768x480_p1_seed0",
        "768x480_p1_seed1",
        "512x320_p1_seed0",
    ]

    every = run(True)
    assert every.count("512x320_p0_seed1") == 1
    assert len(every) == 8


def test_comparing_across_prompts_is_refused(tmp_path):
    """A distance between two prompts measures the prompts, not resolution."""
    pytest.importorskip("PIL")
    import argparse

    cli = _cli()
    reference = _write_run(cli, tmp_path / "768x480_p0_seed0", "a forest path")
    candidate = _write_run(cli, tmp_path / "512x320_p1_seed0", "a city street")

    args = argparse.Namespace(reference=reference, candidate=[candidate], floor=[])
    with pytest.raises(ValueError, match="different prompt"):
        cli._compare(args)


def test_comparing_within_one_prompt_is_allowed(tmp_path):
    cli = _cli()
    reference = _write_run(cli, tmp_path / "768x480_p0_seed0", "a forest path")
    floor = _write_run(cli, tmp_path / "768x480_p0_seed1", "a forest path", seed=1)
    cli._reject_mixed_prompts(reference, [floor])
